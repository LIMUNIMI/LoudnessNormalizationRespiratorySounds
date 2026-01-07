# === CONFIGURATION FILE ===
# --------------------------
# This file contains all the configuration classes used for testing the normalization process.
# Feel free to copy any configuration into the experiment.py to know how every feature influences the results.
# --------------------------

# === Experimenting with Window Sizes ===
    ws_05 = Config(window_size=0.5, result_filename="cross_ws05.csv") # optimal
    ws_06 = Config(window_size=0.6, result_filename="cross_ws06.csv")
    ws_08 = Config(window_size=0.8, result_filename="cross_ws08.csv")
    ws_1 = Config(window_size=1.0, result_filename="cross_ws1.csv")
    ws_15 = Config(window_size=1.5, result_filename="cross_ws15.csv")
    ws_2 = Config(window_size=2.0, result_filename="cross_ws2.csv")
    ws_4 = Config(window_size=4.0, result_filename="cross_ws4.csv")

    # === Experimenting with Duration Normalization ===
    # 1.5 2 2.5 3 3.5
    dt_15 = Config(target_duration=1.5, result_filename="experiment_results_dt1_5.csv")
    dt_2 = Config(target_duration=2.0, result_filename="experiment_results_dt2.csv") # optimal
    dt_25 = Config(target_duration=2.5, result_filename="experiment_results_dt2_5.csv")
    dt_3 = Config(target_duration=3.0, result_filename="experiment_results_dt3.csv")
    dt_35 = Config(target_duration=3.5, result_filename="experiment_results_dt3_5.csv")


    # === Experimenting with Filtering Frequencies ===
    # bp70/2000, hp60, lp1800, bp70/2000+hp60, lp1800+hp60
    bp_70_2000 = Config(bandpass_toggle=True, bandpass_frequency=965.0, bandpass_bandwidth=1930.0, result_filename="experiment_results_bp70_2000.csv") # optimal
    hp_60 = Config(highpass_toggle=True, highpass_frequency=60.0, result_filename="experiment_results_hp60.csv")
    lp_1800 = Config(lowpass_toggle=True, lowpass_frequency=1800.0, result_filename="experiment_results_lp1800.csv")
    bp_70_2000_hp_60 = Config(bandpass_toggle=True, bandpass_frequency=965.0, bandpass_bandwidth=1930.0, highpass_toggle=True, highpass_frequency=60.0, result_filename="experiment_results_bp70_2000_hp60.csv")
    lp_1800_hp_60 = Config(lowpass_toggle=True, lowpass_frequency=1800.0, highpass_toggle=True, highpass_frequency=60.0, result_filename="experiment_results_lp1800_hp60.csv")


    # === Experimenting with Amplitude Normalization ===
    # rms mono, median mono, cluster mono, rms eqloud, median eqloud, cluster eqloud
    # Trying with different cluster numbers: 2, 3
    cluster_2 = Config(n_clusters=2, result_filename="experiment_results_cluster_2ws_08.csv")
    cluster_3 = Config(n_clusters=3, result_filename="experiment_results_cluster_3ws_08.csv")

    # Backtracking with different filtering settings: 2nd order bandpass filter or 4th order bandpass filter
    bp_4th_2clst = Config(n_clusters=2, bandpass_toggle=True, bandpass_frequency=965.0, bandpass_bandwidth=1930.0, fourth_filter_toggle=True, result_filename="4th_filter_2_cluster.csv")
    bp_2nd_2clst = Config(n_clusters=2, bandpass_toggle=True, bandpass_frequency=965.0, bandpass_bandwidth=1930.0, fourth_filter_toggle=False, result_filename="2nd_filter_2_cluster.csv")

    bp_4th_3clst = Config(n_clusters=3, bandpass_toggle=True, bandpass_frequency=965.0, bandpass_bandwidth=1930.0, fourth_filter_toggle=True, result_filename="4th_filter_3_cluster.csv")
    bp_2nd_3clst = Config(n_clusters=3, bandpass_toggle=True, bandpass_frequency=965.0, bandpass_bandwidth=1930.0, fourth_filter_toggle=False, result_filename="2nd_filter_3_cluster.csv")


    # --- Optimization ---
    # Trying with bandpass or without
    bp_on = Config(bandpass_toggle=True, bandpass_frequency=965.0, bandpass_bandwidth=1930.0, fourth_filter_toggle=True, result_filename="bp_on.csv")
    bp_off = Config(bandpass_toggle=False, result_filename="bp_off.csv")

    # Trying with duration normalization or without
    dur_norm_on = Config(duration_norm_toggle=True, result_filename="duration_norm_on.csv")
    dur_norm_off = Config(duration_norm_toggle=False, result_filename="duration_normalization_off.csv")

    # Trying with different splits
    split_1 = Config(result_filename="split_alt_1.csv")
    split_2 = Config(result_filename="split_alt_2.csv")
    split_3 = Config(result_filename="split_alt_3.csv")