use std::{path::Path, sync::Mutex, time::Instant};

use image::RgbImage;
use ndarray::{s, Array4, CowArray, IxDyn};
use ort::{
    tensor::OrtOwnedTensor, Environment, ExecutionProvider, GraphOptimizationLevel, LoggingLevel,
    OrtError, Session, SessionBuilder, Value,
};

type Bbox = [f32; 4]; //[x_top_left, y_top_left, x_bottom_right, y_bottom_right]
type BboxPixels = [u32; 4]; //[x_top_left, y_top_left, x_bottom_right, y_bottom_right]

pub struct UltraPredictor {
    pub name: String,
    pub session: Mutex<Session>,
}

pub struct UltraOutput {
    pub bboxes_with_confidences: Vec<(BboxPixels, f32)>,
}

static CONFIDENCE_THRESHOLD: f32 = 0.5;
static MAX_IOU: f32 = 0.5;
static ULTRA_PREDICTOR_NAME: &str = "UltraPredictor";
pub static ULTRA_INPUT_WIDTH: usize = 640;
pub static ULTRA_INPUT_HEIGHT: usize = 480;
static ULTRA_RATIO: f32 = ULTRA_INPUT_WIDTH as f32 / ULTRA_INPUT_HEIGHT as f32;
static EPS: f32 = 1.0e-7;
/// Positive additive constant to avoid divide-by-zero.

impl UltraPredictor {
    pub fn new(model_filepath: &Path, num_threads: &i16) -> Result<UltraPredictor, OrtError> {
        let start = Instant::now();

        let environment = Environment::builder()
            .with_name(ULTRA_PREDICTOR_NAME.to_string())
            .with_execution_providers([ExecutionProvider::CPU(Default::default())])
            .with_log_level(LoggingLevel::Verbose)
            .build()?
            .into_arc();

        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_intra_threads(*num_threads)?
            .with_model_from_file(&model_filepath)?;

        println!(
            "{} startup took {:?}",
            ULTRA_PREDICTOR_NAME,
            start.elapsed()
        );
        Ok(UltraPredictor {
            name: ULTRA_PREDICTOR_NAME.to_string(),
            session: session.into(),
        })
    }

    pub fn run(&self, image: &RgbImage) -> Result<UltraOutput, OrtError> {
        let start = Instant::now();

        let image_tensor = self.get_image_tensor(&image);
        let image_input = self.get_image_input(&image_tensor)?;
        let raw_outputs = self.session.lock().unwrap().run(image_input)?;
        let bboxes_with_confidences = self.post_process(&raw_outputs)?;
        let ultra_output =
            map_bboxes_to_bbox_with_pixels(image.width(), image.height(), bboxes_with_confidences);

        println!(
            "{} preprocessing and inference took {:?}",
            ULTRA_PREDICTOR_NAME,
            start.elapsed()
        );
        Ok(UltraOutput {
            bboxes_with_confidences: ultra_output,
        })
    }

    fn get_image_tensor(&self, image: &RgbImage) -> CowArray<f32, IxDyn> {
        let image_tensor = CowArray::from(Array4::from_shape_fn(
            (1, 3, ULTRA_INPUT_HEIGHT, ULTRA_INPUT_WIDTH),
            |(_, c, y, x)| {
                let mean = [0.485, 0.456, 0.406][c];
                let std = [0.229, 0.224, 0.225][c];
                (image[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
            },
        ))
        .into_dyn();

        return image_tensor;
    }

    fn get_image_input<'a>(
        &self,
        image_tensor: &'a CowArray<'a, f32, IxDyn>,
    ) -> Result<Vec<Value<'a>>, OrtError> {
        let input_value =
            Value::from_array(self.session.lock().unwrap().allocator(), &image_tensor)?;
        let input = vec![input_value];

        return Ok(input);
    }

    fn post_process(&self, raw_outputs: &Vec<Value>) -> Result<Vec<(Bbox, f32)>, OrtError> {
        let output_0: OrtOwnedTensor<f32, _> = raw_outputs[0].try_extract()?;
        let confidences_view = output_0.view();
        let confidences = confidences_view.slice(s![0, .., 1]);

        let output_1: OrtOwnedTensor<f32, _> = raw_outputs[1].try_extract()?;
        let bbox_view = output_1.view();
        let bbox_arr = bbox_view.to_slice().unwrap().to_vec();
        let bboxes: Vec<Bbox> = bbox_arr.chunks(4).map(|x| x.try_into().unwrap()).collect();

        let mut bboxes_with_confidences: Vec<_> = bboxes
            .iter()
            .zip(confidences.iter())
            .filter_map(|(bbox, confidence)| match confidence {
                x if *x > CONFIDENCE_THRESHOLD => Some((bbox, confidence)),
                _ => None,
            })
            .collect();

        bboxes_with_confidences.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
        let selected_bboxes_with_confidences =
            non_maximum_suppression(bboxes_with_confidences, MAX_IOU).to_vec();

        return Ok(selected_bboxes_with_confidences);
    }
}

/// Run non-maximum-suppression on candidate bounding boxes.
///
/// The pairs of bounding boxes with confidences have to be sorted in **ascending** order of
/// confidence because we want to `pop()` the most confident elements from the back.
///
/// Start with the most confident bounding box and iterate over all other bounding boxes in the
/// order of decreasing confidence. Grow the vector of selected bounding boxes by adding only those
/// candidates which do not have a IoU scores above `max_iou` with already chosen bounding boxes.
/// This iterates over all bounding boxes in `sorted_bboxes_with_confidences`. Any candidates with
/// scores generally too low to be considered should be filtered out before.
fn non_maximum_suppression(
    mut sorted_bboxes_with_confidences: Vec<(&Bbox, &f32)>,
    max_iou: f32,
) -> Vec<(Bbox, f32)> {
    let mut selected = vec![];
    'candidates: loop {
        // Get next most confident bbox from the back of ascending-sorted vector.
        // All boxes fulfill the minimum confidence criterium.
        match sorted_bboxes_with_confidences.pop() {
            Some((bbox, confidence)) => {
                // Check for overlap with any of the selected bboxes
                for (selected_bbox, _) in selected.iter() {
                    match iou(bbox, selected_bbox) {
                        x if x > max_iou => continue 'candidates,
                        _ => (),
                    }
                }

                // bbox has no large overlap with any of the selected ones, add it
                selected.push((*bbox, *confidence))
            }
            None => break 'candidates,
        }
    }

    selected
}

/// Calculate the intersection-over-union metric for two bounding boxes.
fn iou(bbox_a: &Bbox, bbox_b: &Bbox) -> f32 {
    // Calculate corner points of overlap box
    // If the boxes do not overlap, the corner-points will be ill defined, i.e. the top left
    // corner point will be below and to the right of the bottom right corner point. In this case,
    // the area will be zero.
    let overlap_box: Bbox = [
        f32::max(bbox_a[0], bbox_b[0]),
        f32::max(bbox_a[1], bbox_b[1]),
        f32::min(bbox_a[2], bbox_b[2]),
        f32::min(bbox_a[3], bbox_b[3]),
    ];

    let overlap_area = bbox_area(&overlap_box);

    // Avoid division-by-zero with `EPS`
    overlap_area / (bbox_area(bbox_a) + bbox_area(bbox_b) - overlap_area + EPS)
}

/// Calculate the area enclosed by a bounding box.
///
/// The bounding box is passed as four-element array defining two points:
/// `[x_top_left, y_top_left, x_bottom_right, y_bottom_right]`
/// If the bounding box is ill-defined by having the bottom-right point above/to the left of the
/// top-left point, the area is zero.
fn bbox_area(bbox: &Bbox) -> f32 {
    let width = bbox[3] - bbox[1];
    let height = bbox[2] - bbox[0];
    if width < 0.0 || height < 0.0 {
        // bbox is empty/undefined since the bottom-right corner is above the top left corner
        return 0.0;
    }

    width * height
}

fn map_bboxes_to_bbox_with_pixels(
    image_width: u32,
    image_height: u32,
    sorted_bboxes_with_confidences: Vec<(Bbox, f32)>,
) -> Vec<(BboxPixels, f32)> {
    sorted_bboxes_with_confidences
        .into_iter()
        .map(|(bbox, confidence)| {
            let bbox_pixels =
                get_bbox_pixel_locations(image_width as f32, image_height as f32, bbox);
            (bbox_pixels, confidence)
        })
        .collect()
}

fn get_bbox_pixel_locations(image_width: f32, image_height: f32, output_bbox: Bbox) -> BboxPixels {
    let aspect_ratio_raw_image = image_width / image_height;
    let (x_tl, y_tl, x_br, y_br): (f32, f32, f32, f32) = if aspect_ratio_raw_image > ULTRA_RATIO {
        let scaled_width = ULTRA_RATIO * image_height;
        let offset = (image_width - scaled_width) / 2.0;
        (
            output_bbox[0] * scaled_width + offset,
            output_bbox[1] * image_height,
            output_bbox[2] * scaled_width + offset,
            output_bbox[3] * image_height,
        )
    } else if aspect_ratio_raw_image < ULTRA_RATIO {
        let scaled_height = (1.0 / ULTRA_RATIO) * image_width;
        let offset = (image_height - scaled_height) / 2.0;
        (
            output_bbox[0] * image_width,
            output_bbox[1] * scaled_height + offset,
            output_bbox[2] * image_width,
            output_bbox[3] * scaled_height + offset,
        )
    } else {
        // raw_image has same aspect ratio
        (
            output_bbox[0] * image_width,
            output_bbox[1] * image_height,
            output_bbox[2] * image_width,
            output_bbox[3] * image_height,
        )
    };
    [x_tl as u32, y_tl as u32, x_br as u32, y_br as u32]
}
