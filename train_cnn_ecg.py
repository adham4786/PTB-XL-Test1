import os
import ast
import argparse
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, resample
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import json


def parse_args():
    p = argparse.ArgumentParser(description='Train 1D CNN for AF detection on PTB-XL')
    p.add_argument('--csv', default='data/ptbxl_database.csv')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--target-samples', type=int, default=5000, help='Number of samples per recording (truncate/pad)')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--save-model', default='af_cnn.h5')
    return p.parse_args()


def parse_scp_codes(s):
    # scp_codes column stores a string representation of a dict, e.g. "{'NORM':100.0,'AFIB':0.0}"
    if pd.isna(s) or str(s).strip() == '':
        return {}
    try:
        d = ast.literal_eval(s)
        if isinstance(d, dict):
            return d
    except Exception:
        # fallback: try to extract keys heuristically
        try:
            items = s.strip().strip('"').strip("'")
            # naive split
            keys = [seg.split(':')[0].strip(" '\"{}") for seg in items.split(',') if ':' in seg]
            return {k: 1.0 for k in keys}
        except Exception:
            return {}
    return d


def is_af(scp_dict):
    # consider different tokens for atrial fibrillation/flutter
    if not scp_dict:
        return 0
    keys = [k.upper() for k in scp_dict.keys()]
    for k in keys:
        if 'AF' in k or 'AFL' in k or 'AFIB' in k or 'AFLT' in k:
            return 1
    return 0


def load_record(path_without_ext, target_samples):
    # path_without_ext: full path to file without .dat/.hea extension
    try:
        rec = wfdb.rdrecord(path_without_ext)
    except Exception as e:
        print(f"Failed to read {path_without_ext}: {e}")
        return None
    sig = rec.p_signal  # shape (n_samples, n_leads)
    if sig is None:
        return None
    # optional: bandpass filter
    sig = bandpass(sig, rec.fs, low=0.5, high=40.0)
    # resample to target_samples
    n_samples = sig.shape[0]
    if n_samples != target_samples:
        sig = resample_to_length(sig, target_samples)
    # normalize per-lead
    sig = (sig - np.mean(sig, axis=0)) / (np.std(sig, axis=0) + 1e-8)
    return sig.astype(np.float32)


def bandpass(sig, fs, low=0.5, high=40.0, order=4):
    # sig shape (n_samples, n_leads)
    nyq = 0.5 * fs
    lowb = low / nyq
    highb = high / nyq
    try:
        b, a = butter(order, [lowb, highb], btype='band')
        filtered = filtfilt(b, a, sig, axis=0)
        return filtered
    except Exception:
        return sig


def resample_to_length(sig, target_len):
    n_samples, n_leads = sig.shape
    # resample each lead
    res = resample(sig, target_len, axis=0)
    return res


def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    # example 1D CNN stack
    x = layers.Conv1D(32, kernel_size=7, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)

    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)

    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    # Use explicit metric objects to ensure stable metric names across TF versions
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')]
    )
    return model


def generator(filepaths, labels, batch_size, target_samples, data_dir):
    n = len(filepaths)
    idxs = np.arange(n)
    while True:
        np.random.shuffle(idxs)
        for i in range(0, n, batch_size):
            batch_idx = idxs[i:i+batch_size]
            Xb = []
            yb = []
            for j in batch_idx:
                fp = filepaths[j]
                # full path without extension
                base = os.path.join(data_dir, fp)
                base = base.strip().strip('\"').strip("'")
                rec = os.path.join(base)
                sig = load_record(rec, target_samples)
                if sig is None:
                    continue
                Xb.append(sig)
                yb.append(labels[j])
            if len(Xb) == 0:
                continue
            Xb = np.stack(Xb, axis=0)
            yield Xb, np.array(yb, dtype=np.float32)


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    # parse scp_codes for AF label
    df['scp_dict'] = df['scp_codes'].apply(parse_scp_codes)
    df['af_label'] = df['scp_dict'].apply(is_af)

    # use the HR filename column
    if 'filename_hr' in df.columns:
        df['file_path'] = df['filename_hr'].astype(str)
    else:
        # try filename_lr fallback
        df['file_path'] = df['filename_lr'].astype(str)

    # filter rows where file exists
    def check_exists(fp):
        base = os.path.join(args.data_dir, fp)
        # files have .dat and .hea
        return os.path.exists(base + '.dat') or os.path.exists(base + '.hea')

    df['exists'] = df['file_path'].apply(check_exists)
    df = df[df['exists']]
    if df.shape[0] == 0:
        raise RuntimeError('No records found. Check --data-dir and CSV filenames.')

    X_files = df['file_path'].tolist()
    y = df['af_label'].values

    # split
    X_train, X_val, y_train, y_val = train_test_split(X_files, y, test_size=0.2, random_state=42, stratify=y)

    sample_sig = None
    # try to find one sample to determine shape
    for fp in X_train:
        base = os.path.join(args.data_dir, fp).strip()
        try:
            rec = wfdb.rdrecord(base)
            sample_sig = rec.p_signal
            break
        except Exception:
            continue
    if sample_sig is None:
        raise RuntimeError('Unable to read any record to infer channels.')
    n_leads = sample_sig.shape[1]
    input_shape = (args.target_samples, n_leads)

    model = build_model(input_shape)
    model.summary()

    train_gen = generator(X_train, y_train, args.batch_size, args.target_samples, args.data_dir)
    val_gen = generator(X_val, y_val, args.batch_size, args.target_samples, args.data_dir)

    steps_per_epoch = max(1, int(len(X_train) / args.batch_size))
    validation_steps = max(1, int(len(X_val) / args.batch_size))

    cb = [callbacks.ModelCheckpoint(args.save_model, save_best_only=True, monitor='val_auc', mode='max'),
          callbacks.EarlyStopping(patience=5, monitor='val_auc', mode='max', restore_best_weights=True)]

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=cb,
    )

    model.save(args.save_model)
    print(f"Saved model to {args.save_model}")

    # training history is a dict: history.history (keys: loss, accuracy, val_loss, val_accuracy, auc, ...)
    hist_dict = getattr(history, 'history', {})
    # evaluate on validation set and get a dict of metric -> value (if available)
    eval_results = None
    if hasattr(model, 'evaluate'):
        try:
            # Keras evaluate can return a dict when return_dict=True
            eval_results = model.evaluate(val_gen, steps=validation_steps, return_dict=True)
        except TypeError:
            # older TF may not support return_dict; fall back to list
            vals = model.evaluate(val_gen, steps=validation_steps)
            # map names to values if possible
            names = model.metrics_names if hasattr(model, 'metrics_names') else []
            eval_results = dict(zip(names, vals))

    # save history and eval results for inspection
    out = {
        'history': hist_dict,
        'eval_results': eval_results,
    }
    try:
        with open(args.save_model + '.training_history.json', 'w') as f:
            json.dump(out, f, indent=2, default=lambda o: str(o))
        print(f"Wrote training history and eval results to {args.save_model}.training_history.json")
    except Exception as e:
        print(f"Could not write training history: {e}")


if __name__ == '__main__':
    main()
