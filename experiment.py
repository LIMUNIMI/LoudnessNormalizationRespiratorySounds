import os
from sklearn.preprocessing import StandardScaler

from config import *
from file_utils import get_segments, split_train_test_dirs, get_labels_all_dirs, read_labels, split_train_test_files, save_results
from normalization_utils import parallel_normalize_all_segments_duration, parallel_normalize_all_files, parallel_normalize_all_files_cluster
from filters import parallel_filter_all_files
from clustering import compute_cluster_features, fit_kmeans
from features import extract_all_features
from classification import evaluate

import logging
import gc


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler()
    ]
)
lg = logging.getLogger(__name__)
cfg = Config()

def main(cfg: Config):


    lg.info("\n\n")
    lg.info("===================================")
    lg.info("Configuring experiment...")


    dataset_directory = "resampled_data/"
    segmentation_directory = f'{dataset_directory}segments/'
    duration_norm_directory = "duration_norm_data/"
    filtering_directory = "filtered_data/"
    amplitude_norm_directory = "amplitude_norm_data/"
    metadata_directory = "metadata/"
    seg_data_directory = "seg_data/"
    results_directory = "results/"


    # === Segmentation ===
    lg.info("Starting segmentation...")
    get_segments(
        source_dir=dataset_directory,
        destination_dir=segmentation_directory,
        data_dir=seg_data_directory)
    

    # === Duration Normalization ===
    if cfg.duration_norm_toggle:
        lg.info("Starting duration normalization...")
        parallel_normalize_all_segments_duration(
            source_dir=segmentation_directory,
            output_dir=duration_norm_directory,
            sample_rate=cfg.sample_rate,
            target_length=cfg.target_duration)
    else:
        lg.info("Skipping duration normalization...")
        duration_norm_directory = segmentation_directory


    # === Filtering ===
    if cfg.highpass_toggle or cfg.bandpass_toggle:
        lg.info("Starting filtering...")
        parallel_filter_all_files(
            source_dir=duration_norm_directory,
            output_dir=filtering_directory,
            sample_rate=cfg.sample_rate,
            use_hp=cfg.highpass_toggle,
            hp_cutoff=cfg.highpass_frequency,
            use_bp=cfg.bandpass_toggle,
            bp_cutoff=cfg.bandpass_frequency,
            bp_bandwidth=cfg.bandpass_bandwidth)
    else:
        lg.info("Skipping filtering...")
        filtering_directory = duration_norm_directory
    

    # === Amplitude Normalization ===
    if cfg.amplitude_norm_toggle:
        lg.info("Starting amplitude normalization...")
        # === Mono Loading ===
        lg.info("Loading mono files...")


        rms_mono_dir = f'{amplitude_norm_directory}mono/rms/'
        parallel_normalize_all_files(
            source_dir=filtering_directory,
            output_dir=rms_mono_dir,
            sample_rate=cfg.sample_rate,
            method='rms',
            loader='mono'
        )

        median_mono_dir = f'{amplitude_norm_directory}mono/median/'
        parallel_normalize_all_files(
            source_dir=filtering_directory,
            output_dir=median_mono_dir,
            sample_rate=cfg.sample_rate,
            method='median',
            loader='mono'
        )

        cluster_mono_dir = f'{amplitude_norm_directory}mono/cluster/'
        clustering_features_mono = compute_cluster_features(
            directory=filtering_directory,
            loader_fn='mono',
            sample_rate=cfg.sample_rate,
            use_hp=cfg.highpass_toggle,
            use_bp=cfg.bandpass_toggle,
            hp_cutoff=cfg.highpass_frequency,
            bp_cutoff=cfg.bandpass_frequency,
            bp_bandwidth=cfg.bandpass_bandwidth,
            features=cfg.cluster_features
        )
        km = fit_kmeans(
            features=clustering_features_mono,
            n_clusters=cfg.n_clusters,
            random_state=cfg.random_state
        )
        scaler = StandardScaler()
        scaled_features_mono = scaler.fit_transform(clustering_features_mono)
        parallel_normalize_all_files_cluster(
            source_dir=filtering_directory,
            output_dir=cluster_mono_dir,
            sample_rate=cfg.sample_rate,
            km=km,
            scaled_features=scaled_features_mono,
            loader='mono'
        )


        # === EqLoud Loading ===
        lg.info("Loading EqLoud files...")


        rms_eqloud_dir = f'{amplitude_norm_directory}eqloud/rms/'
        parallel_normalize_all_files(
            source_dir=filtering_directory,
            output_dir=rms_eqloud_dir,
            sample_rate=cfg.sample_rate,
            method='rms',
            loader='eqloud'
        )

        median_eqloud_dir = f'{amplitude_norm_directory}eqloud/median/'
        parallel_normalize_all_files(
            source_dir=filtering_directory,
            output_dir=median_eqloud_dir,
            sample_rate=cfg.sample_rate,
            method='median',
            loader='eqloud'
        )

        cluster_eqloud_dir = f'{amplitude_norm_directory}eqloud/cluster/'
        clustering_features_eqloud = compute_cluster_features(
            directory=filtering_directory,
            loader_fn='eqloud',
            sample_rate=cfg.sample_rate,
            use_hp=cfg.highpass_toggle,
            use_bp=cfg.bandpass_toggle,
            hp_cutoff=cfg.highpass_frequency,
            bp_cutoff=cfg.bandpass_frequency,
            bp_bandwidth=cfg.bandpass_bandwidth,
            features=cfg.cluster_features
        )
        km = fit_kmeans(
            features=clustering_features_eqloud,
            n_clusters=cfg.n_clusters,
            random_state=cfg.random_state
        )
        scaler = StandardScaler()
        scaled_features_eqloud = scaler.fit_transform(clustering_features_eqloud)
        parallel_normalize_all_files_cluster(
            source_dir=filtering_directory,
            output_dir=cluster_eqloud_dir,
            sample_rate=cfg.sample_rate,
            km=km,
            scaled_features=scaled_features_eqloud,
            loader='eqloud'
        )
        lg.info("Amplitude normalization completed.")
    else:
        lg.info("Skipping amplitude normalization...")
        amplitude_norm_directory = filtering_directory


    # === Memory Cleanup ===
    lg.info("Cleaning up memory...")
    gc.collect()


    # === Splitting Train/Test Sets & Generating Labels ===
    lg.info("Splitting train and test sets...")
    split_file_path = f'{metadata_directory}ICBHI_challenge_train_test.txt'
    label_file_path = f'{metadata_directory}ICBHI_challenge_diagnosis.txt'
    if cfg.amplitude_norm_toggle:
        split_train_test_dirs(
            source_dir=f'{amplitude_norm_directory}mono/',
            split_file_path=split_file_path
        )

        split_train_test_dirs(
            source_dir=f'{amplitude_norm_directory}eqloud/',
            split_file_path=split_file_path
        )


        lg.info("Generating labels...")
        get_labels_all_dirs(
            source_dir=rms_mono_dir,
            label_file_path=label_file_path
        )
        get_labels_all_dirs(
            source_dir=median_mono_dir,
            label_file_path=label_file_path
        )
        get_labels_all_dirs(
            source_dir=cluster_mono_dir,
            label_file_path=label_file_path
        )
        get_labels_all_dirs(
            source_dir=rms_eqloud_dir,
            label_file_path=label_file_path
        )
        get_labels_all_dirs(
            source_dir=median_eqloud_dir,
            label_file_path=label_file_path
        )
        get_labels_all_dirs(
            source_dir=cluster_eqloud_dir,
            label_file_path=label_file_path
        )
    else:
        split_train_test_files(
            source_dir=amplitude_norm_directory,
            train_dir=f'{amplitude_norm_directory}train/',
            test_dir=f'{amplitude_norm_directory}test/',
            split_file_path=split_file_path
        )
        get_labels_all_dirs(
            source_dir=amplitude_norm_directory,
            label_file_path=label_file_path
        )


    # === Feature Extraction & Classification ===
    lg.info("Extracting features and classifying...")
    results = {}
    if cfg.amplitude_norm_toggle:
        rms_mono_train = extract_all_features(source_dir=f'{rms_mono_dir}train/', cfg=cfg)
        rms_mono_test = extract_all_features(source_dir=f'{rms_mono_dir}test/', cfg=cfg)
        results['rms_mono'] = evaluate(rms_mono_train, read_labels(f'{rms_mono_dir}train/labels.csv'), rms_mono_test, read_labels(f'{rms_mono_dir}test/labels.csv'), cfg)


        median_mono_train = extract_all_features(source_dir=f'{median_mono_dir}train/', cfg=cfg)
        median_mono_test = extract_all_features(source_dir=f'{median_mono_dir}test/', cfg=cfg)
        results['median_mono'] = evaluate(median_mono_train, read_labels(f'{median_mono_dir}train/labels.csv'), median_mono_test, read_labels(f'{median_mono_dir}test/labels.csv'), cfg)

        cluster_mono_train = extract_all_features(source_dir=f'{cluster_mono_dir}train/', cfg=cfg)
        cluster_mono_test = extract_all_features(source_dir=f'{cluster_mono_dir}test/', cfg=cfg)
        results['cluster_mono'] = evaluate(cluster_mono_train, read_labels(f'{cluster_mono_dir}train/labels.csv'), cluster_mono_test, read_labels(f'{cluster_mono_dir}test/labels.csv'), cfg)

        rms_eqloud_train = extract_all_features(source_dir=f'{rms_eqloud_dir}train/', cfg=cfg)
        rms_eqloud_test = extract_all_features(source_dir=f'{rms_eqloud_dir}test/', cfg=cfg)
        results['rms_eqloud'] = evaluate(rms_eqloud_train, read_labels(f'{rms_eqloud_dir}train/labels.csv'), rms_eqloud_test, read_labels(f'{rms_eqloud_dir}test/labels.csv'), cfg)

        median_eqloud_train = extract_all_features(source_dir=f'{median_eqloud_dir}train/', cfg=cfg)
        median_eqloud_test = extract_all_features(source_dir=f'{median_eqloud_dir}test/', cfg=cfg)
        results['median_eqloud'] = evaluate(median_eqloud_train, read_labels(f'{median_eqloud_dir}train/labels.csv'), median_eqloud_test, read_labels(f'{median_eqloud_dir}test/labels.csv'), cfg)

        cluster_eqloud_train = extract_all_features(source_dir=f'{cluster_eqloud_dir}train/', cfg=cfg)
        cluster_eqloud_test = extract_all_features(source_dir=f'{cluster_eqloud_dir}test/', cfg=cfg)
        results['cluster_eqloud'] = evaluate(cluster_eqloud_train, read_labels(f'{cluster_eqloud_dir}train/labels.csv'), cluster_eqloud_test, read_labels(f'{cluster_eqloud_dir}test/labels.csv'), cfg)
    else:
        all_train = extract_all_features(source_dir=f'{amplitude_norm_directory}train/', cfg=cfg)
        all_test = extract_all_features(source_dir=f'{amplitude_norm_directory}test/', cfg=cfg)
        results['all'] = evaluate(all_train, read_labels(f'{amplitude_norm_directory}train/labels.csv'), all_test, read_labels(f'{amplitude_norm_directory}test/labels.csv'), cfg)


    # === Results Logging ===
    lg.info("Saving results to CSV...")
    save_results(results, os.path.join(results_directory, cfg.result_filename))
    lg.info("Experiment Results:")
    for method, metrics in results.items():
        lg.info(f"Method: {method}")
        for metric_name, metric_value in metrics.items():
            lg.info(f"  {metric_name}: {metric_value:.4f}")


def main_eval_only():


    lg.info("\n\n")
    lg.info("===================================")
    lg.info("Configuring experiment...")


    dataset_directory = "resampled_data/"
    segmentation_directory = f'{dataset_directory}segments/'
    duration_norm_directory = "duration_norm_data/"
    filtering_directory = "filtered_data/"
    amplitude_norm_directory = "amplitude_norm_data/"
    metadata_directory = "metadata/"
    seg_data_directory = "seg_data/"
    rms_mono_dir = f'{amplitude_norm_directory}mono/rms/'
    median_mono_dir = f'{amplitude_norm_directory}mono/median/'
    cluster_mono_dir = f'{amplitude_norm_directory}mono/cluster/'
    rms_eqloud_dir = f'{amplitude_norm_directory}eqloud/rms/'
    median_eqloud_dir = f'{amplitude_norm_directory}eqloud/median/'
    cluster_eqloud_dir = f'{amplitude_norm_directory}eqloud/cluster/'


    # === Feature Extraction & Classification ===
    lg.info("Extracting features and classifying...")
    results = {}

    rms_mono_train = extract_all_features(source_dir=f'{rms_mono_dir}train/', cfg=cfg)
    rms_mono_test = extract_all_features(source_dir=f'{rms_mono_dir}test/', cfg=cfg)
    results['rms_mono'] = evaluate(rms_mono_train, read_labels(f'{rms_mono_dir}train/labels.csv'), rms_mono_test, read_labels(f'{rms_mono_dir}test/labels.csv',), cfg)


    median_mono_train = extract_all_features(source_dir=f'{median_mono_dir}train/', cfg=cfg)
    median_mono_test = extract_all_features(source_dir=f'{median_mono_dir}test/', cfg=cfg)
    results['median_mono'] = evaluate(median_mono_train, read_labels(f'{median_mono_dir}train/labels.csv'), median_mono_test, read_labels(f'{median_mono_dir}test/labels.csv'), cfg)

    cluster_mono_train = extract_all_features(source_dir=f'{cluster_mono_dir}train/', cfg=cfg)
    cluster_mono_test = extract_all_features(source_dir=f'{cluster_mono_dir}test/', cfg=cfg)
    results['cluster_mono'] = evaluate(cluster_mono_train, read_labels(f'{cluster_mono_dir}train/labels.csv'), cluster_mono_test, read_labels(f'{cluster_mono_dir}test/labels.csv'), cfg)

    rms_eqloud_train = extract_all_features(source_dir=f'{rms_eqloud_dir}train/', cfg=cfg)
    rms_eqloud_test = extract_all_features(source_dir=f'{rms_eqloud_dir}test/', cfg=cfg)
    results['rms_eqloud'] = evaluate(rms_eqloud_train, read_labels(f'{rms_eqloud_dir}train/labels.csv'), rms_eqloud_test, read_labels(f'{rms_eqloud_dir}test/labels.csv'), cfg)

    median_eqloud_train = extract_all_features(source_dir=f'{median_eqloud_dir}train/', cfg=cfg)
    median_eqloud_test = extract_all_features(source_dir=f'{median_eqloud_dir}test/', cfg=cfg)
    results['median_eqloud'] = evaluate(median_eqloud_train, read_labels(f'{median_eqloud_dir}train/labels.csv'), median_eqloud_test, read_labels(f'{median_eqloud_dir}test/labels.csv'), cfg)

    cluster_eqloud_train = extract_all_features(source_dir=f'{cluster_eqloud_dir}train/', cfg=cfg)
    cluster_eqloud_test = extract_all_features(source_dir=f'{cluster_eqloud_dir}test/', cfg=cfg)
    results['cluster_eqloud'] = evaluate(cluster_eqloud_train, read_labels(f'{cluster_eqloud_dir}train/labels.csv'), cluster_eqloud_test, read_labels(f'{cluster_eqloud_dir}test/labels.csv'), cfg)


    # === Results Logging ===
    lg.info("Experiment Results:")
    for method, metrics in results.items():
        lg.info(f"Method: {method}")
        for metric_name, metric_value in metrics.items():
            lg.info(f"  {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    cfg_05 = Config(window_size=0.5, result_filename="experiment_results_ws05.csv")
    cfg_06 = Config(window_size=0.6, result_filename="experiment_results_ws06.csv")
    cfg_075 = Config(window_size=0.75, result_filename="experiment_results_ws075.csv")
    cfg_08 = Config(window_size=0.8, result_filename="experiment_results_ws08.csv")
    cfg_1 = Config(window_size=1.0, result_filename="experiment_results_ws1.csv")
    cfg_125 = Config(window_size=1.25, result_filename="experiment_results_ws125.csv")
    cfg_15 = Config(window_size=1.5, result_filename="experiment_results_ws15.csv")
    cfg_2 = Config(window_size=2.0, result_filename="experiment_results_ws2.csv")
    cfg_4 = Config(window_size=4.0, result_filename="experiment_results_ws4.csv")


    if cfg.run_method == "all":
        main(cfg=cfg_05)
        main(cfg=cfg_06)
        main(cfg=cfg_075)
        main(cfg=cfg_08)
        main(cfg=cfg_1)
        main(cfg=cfg_125)
        main(cfg=cfg_15)
        main(cfg=cfg_2)
        main(cfg=cfg_4)
    elif cfg.run_method == "classification":
        main_eval_only()
