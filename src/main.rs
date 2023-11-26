use actix_files;
use actix_multipart::form::{tempfile::TempFile, MultipartForm};
use actix_web::{
    post,
    web::{self},
    App, HttpResponse, HttpServer, Responder,
};
use image::ImageFormat;
use mime;
use std::fs;

use face_detection_server::{
    config::Config, image_queue::ImageQueue, queue_processor::process_queue_task,
    ultra_predictor::UltraPredictor,
};
use serde::{Deserialize, Serialize};
use std::{process, sync::Arc};

#[derive(MultipartForm)]
pub struct Upload {
    #[multipart(limit = "20 MiB")]
    file: TempFile,
}

struct AppState {
    queue: Arc<ImageQueue>,
}

#[derive(Serialize, Deserialize)]
struct QueueResponse {
    id: Option<String>,
    err: Option<String>,
}

#[post("/queue")]
async fn add_to_queue(
    file_payload: MultipartForm<Upload>,
    data: web::Data<AppState>,
) -> impl Responder {
    let temp_file = file_payload.0.file;
    let content_type_opt = temp_file.content_type;

    let content_type = match content_type_opt {
        Some(content) => content,
        None => {
            return HttpResponse::BadRequest().json(QueueResponse {
                id: None,
                err: Some("content_type not specified".to_string()),
            });
        }
    };

    match (content_type.type_(), content_type.subtype()) {
        (mime::IMAGE, mime::PNG) => {}
        (mime::IMAGE, mime::JPEG) => {}
        _ => {
            return HttpResponse::BadRequest().json(QueueResponse {
                id: None,
                err: Some("content_type not supported".to_string()),
            });
        }
    };

    let format = match ImageFormat::from_mime_type(content_type) {
        Some(format) => format,
        None => {
            return HttpResponse::BadRequest().json(QueueResponse {
                id: None,
                err: Some("unable to find image format for content_type".to_string()),
            });
        }
    };

    if temp_file.size < 1 {
        let _ = temp_file.file.close();
        return HttpResponse::BadRequest().json(QueueResponse {
            id: None,
            err: Some("file size is 0".to_string()),
        });
    }

    if data.queue.is_full() {
        let _ = temp_file.file.close();
        return HttpResponse::ServiceUnavailable().json(QueueResponse {
            id: None,
            err: Some("queue is full".to_string()),
        });
    }

    let (_, path) = match temp_file.file.keep() {
        Ok(res) => res,
        Err(_) => {
            return HttpResponse::InternalServerError().json(QueueResponse {
                id: None,
                err: Some("could not store file".to_string()),
            });
        }
    };

    let id = data.queue.push(path, format);

    return HttpResponse::Created().json(QueueResponse {
        id: Some(id.to_string()),
        err: None,
    });
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config: Config = Config::new();
    let ultra_predictor = Arc::new(
        UltraPredictor::new(&config.ultra_model_path, &config.ultra_threads).unwrap_or_else(
            |ort_err| {
                println!(
                    "Problem creating ultra onnx session: {}",
                    ort_err.to_string()
                );
                process::exit(1)
            },
        ),
    );
    let queue = Arc::new(ImageQueue::new());

    let app_state = web::Data::new(AppState {
        queue: queue.clone(),
    });

    let _ = fs::create_dir("./results");

    actix_rt::spawn(
        async move { process_queue_task(ultra_predictor.clone(), queue.clone()).await },
    );

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(add_to_queue)
            .service(actix_files::Files::new("/result", "./results"))
    })
    .bind(("127.0.0.1", 8082))?
    .run()
    .await
}
