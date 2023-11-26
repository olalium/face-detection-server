use dotenv::dotenv;
use std::{env, path::PathBuf, process};

pub struct Config {
    pub ultra_model_path: PathBuf,
    pub ultra_threads: i16,
}

impl Config {
    pub fn new() -> Config {
        dotenv().ok();
        let ultra_model_path = env::var("ULTRA_MODEL_PATH").unwrap_or_else(|err| {
            println!("Unable to get ULTRA_MODEL_PATH env variable: {}", err);
            process::exit(1);
        });
        let ultra_model_path = PathBuf::from(&ultra_model_path);
        if !ultra_model_path.exists() {
            println!("Unable to find ULTRA_MODEL_PATH");
            process::exit(1);
        }

        let ultra_threads: i16 = env::var("ULTRA_THREADS")
            .unwrap_or_else(|err| {
                println!("Unable to get ULTRA_THREADS env variable: {}", err);
                process::exit(1);
            })
            .parse()
            .unwrap_or_else(|err| {
                println!("Unable to parse ULTRA_THREADS env variable: {}", err);
                process::exit(1)
            });

        Config {
            ultra_model_path: ultra_model_path,
            ultra_threads,
        }
    }
}
