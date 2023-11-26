use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process,
    sync::Arc,
    time::Duration,
};

use actix_rt::time;
use image::{io::Reader, imageops::FilterType};

use crate::{image_queue::ImageQueue, ultra_predictor::{UltraPredictor, ULTRA_INPUT_WIDTH, ULTRA_INPUT_HEIGHT}};

static POLL_INTERVAL_MS: u64 = 10;

pub async fn process_queue_task(ultra_predictor: Arc<UltraPredictor>, queue: Arc<ImageQueue>) {
    let mut interval = time::interval(Duration::from_millis(POLL_INTERVAL_MS));

    loop {
        interval.tick().await;
        for item in queue.drain() {
            let image_location = item.image_location;

            let mut image_buf = match Reader::open(image_location.clone()) {
                Ok(image_buf) => image_buf,
                Err(_) => {
                    println!("Unable to open_image");
                    remove_temp_file(image_location.clone());
                    continue;
                }
            };

            image_buf.set_format(item.format);
            let raw_image = match image_buf.decode() {
                Ok(raw_image) => raw_image,
                Err(_) => {
                    println!("unable to decode image");
                    remove_temp_file(image_location.clone());
                    continue;
                }
            };

            let image = raw_image.resize_to_fill(
                ULTRA_INPUT_WIDTH as u32,
                ULTRA_INPUT_HEIGHT as u32,
                FilterType::Triangle).to_rgb8();

            let res = ultra_predictor.run(&image).unwrap();

            let results_folder = Path::new("./results");
            let file = match File::create(results_folder.join(item.id.to_string() + ".json")) {
                Ok(file) => file,
                Err(_) => {
                    println!("unable to create result file");
                    remove_temp_file(image_location.clone());
                    continue;
                }
            };

            let mut writer = BufWriter::new(file);

            // TODO: also store some more info about the processing-job
            match serde_json::to_writer(&mut writer, &res.bboxes_with_confidences) {
                Ok(_) => {}
                Err(_) => {
                    println!("unable to write result");
                    remove_temp_file(image_location.clone());
                    continue;
                }
            };

            match writer.flush() {
                Ok(_) => {}
                Err(_) => {
                    println!("unable to write result");
                    remove_temp_file(image_location.clone());
                    continue;
                }
            };

            remove_temp_file(image_location.clone())
        }
    }
}

fn remove_temp_file(image_location: PathBuf) {
    println!("deleting temp file, {}", image_location.to_string_lossy());
    match fs::remove_file(image_location) {
        Ok(_) => {}
        Err(err) => {
            println!("[FATAL] unable to remove temp file; {}", err.to_string());
            process::exit(-1)
        }
    }
}
