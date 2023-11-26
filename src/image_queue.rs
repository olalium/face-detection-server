use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::SystemTime,
};

use image::ImageFormat;
use uuid::Uuid;

static QUEUE_SIZE: usize = 10000;

pub struct QueueItem {
    pub id: Uuid,
    pub image_location: PathBuf,
    pub format: ImageFormat,
    pub added_time: SystemTime,
}

pub struct ImageQueue {
    pub queue: Arc<Mutex<Vec<QueueItem>>>,
}

impl ImageQueue {
    pub fn new() -> ImageQueue {
        let queue = Arc::new(Mutex::new(Vec::with_capacity(QUEUE_SIZE)));
        ImageQueue { queue }
    }

    pub fn drain(&self) -> Vec<QueueItem> {
        let queue_items: Vec<QueueItem> = { self.queue.lock().unwrap().drain(..).collect() };
        queue_items
    }

    pub fn is_full(&self) -> bool {
        self.queue.lock().unwrap().len() > QUEUE_SIZE
    }

    pub fn push(&self, image_location: PathBuf, format: ImageFormat) -> Uuid {
        let id = Uuid::new_v4();
        self.queue.lock().unwrap().push(QueueItem {
            id,
            image_location,
            format,
            added_time: SystemTime::now(),
        });
        return id;
    }
}
