use std::sync::Arc;

use burn::{data::dataloader::DataLoader, module::AutodiffModule, optim::Optimizer, record::CompactRecorder, tensor::{backend::AutodiffBackend, Device}, train::{metric::{store::{Aggregate, Direction, Split}, AccuracyMetric, CpuMemory, CpuTemperature, CpuUse, LossMetric}, ClassificationOutput, LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition, TrainStep, ValidStep}};

use crate::batcher::ImageClassificationBatch;

pub fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn basic_train<
    B: AutodiffBackend,
    M: AutodiffModule<B> + std::fmt::Display + TrainStep<ImageClassificationBatch<B>, ClassificationOutput<B>> + ValidStep<ImageClassificationBatch<B>, ClassificationOutput<B>> + 'static,
    O: Optimizer<M, B> + 'static> (
    artifact_dir: &str,
    device: &Device<B>,
    epochs: usize,
    learning_rate: f64,
    model: M,
    optimizer: O,
    train: Arc<(dyn DataLoader<ImageClassificationBatch<B>> + 'static)>,
    valid: Arc<(dyn DataLoader<ImageClassificationBatch<B::InnerBackend>> + 'static)>,
) -> M where M::InnerModule: ValidStep<ImageClassificationBatch<B::InnerBackend>, ClassificationOutput<B::InnerBackend>> {
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(epochs)
        .summary()
        .build(model, optimizer, learning_rate);

    learner.fit(train, valid)
}