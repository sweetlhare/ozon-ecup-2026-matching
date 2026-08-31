#!/usr/bin/env python3
"""Строгая проверка submission ZIP перед ручной отправкой.

Работает только на стандартной библиотеке: проверяет CRC всех файлов без
распаковки, обязательный runtime, структуру моделей, metadata и заголовки
опциональных lookup-массивов. Завершается ненулевым кодом при любой ошибке.
"""
import argparse
import ast
import hashlib
import json
import math
import posixpath
import struct
import sys
import zipfile


RUNTIME = {
    "run.py", "metadata.json", "pair_text.py", "attr_canon.py",
    "brands.json", "pri_by_cat.json",
}
FORBIDDEN = {"out.csv", ".DS_Store"}


def npy_header(fh):
    if fh.read(6) != b"\x93NUMPY":
        raise ValueError("не NPY-файл")
    major, minor = fh.read(2)
    if major == 1:
        size = struct.unpack("<H", fh.read(2))[0]
    elif major in (2, 3):
        size = struct.unpack("<I", fh.read(4))[0]
    else:
        raise ValueError(f"неподдерживаемая версия NPY {major}.{minor}")
    header = ast.literal_eval(fh.read(size).decode("latin1").strip())
    return header["descr"], tuple(header["shape"]), bool(header["fortran_order"])


def safetensors_header(fh, file_size):
    raw = fh.read(8)
    if len(raw) != 8:
        raise ValueError("обрезан 8-байтовый header size")
    size = struct.unpack("<Q", raw)[0]
    if size <= 2 or size > file_size - 8:
        raise ValueError(f"невозможный header size {size}")
    header = json.loads(fh.read(size))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        raise ValueError("нет тензоров")
    payload = file_size - 8 - size
    for name, spec in tensors.items():
        if not isinstance(spec, dict) or not {"dtype", "shape", "data_offsets"} <= set(spec):
            raise ValueError(f"{name}: неполное описание")
        start, end = spec["data_offsets"]
        if not (0 <= start <= end <= payload):
            raise ValueError(f"{name}: data_offsets вне файла")
    return len(tensors), sorted({spec["dtype"] for spec in tensors.values()})


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(errors, message):
    errors.append(message)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--expect-models", type=int, default=0,
                    help="точное число каталогов model*; 0 = хотя бы один")
    ap.add_argument("--expect-ordered", nargs="*", default=[], metavar="MODEL_DIR")
    ap.add_argument("--expect-lookup", action="store_true")
    ap.add_argument("--expect-closure", action="store_true")
    ap.add_argument("--expect-textsig", action="store_true",
                    help="ожидать negative-consensus по точным ORDERED-текстам")
    ap.add_argument("--require-fp16", action="store_true",
                    help="запретить FP32/FP64 floating tensors в весах")
    args = ap.parse_args()

    errors = []
    try:
        zf = zipfile.ZipFile(args.archive)
    except (OSError, zipfile.BadZipFile) as exc:
        print(f"ОШИБКА: архив не открывается: {exc}", file=sys.stderr)
        return 1

    with zf:
        infos = zf.infolist()
        names = [i.filename for i in infos]
        name_set = set(names)
        if len(names) != len(name_set):
            fail(errors, "в ZIP есть повторяющиеся имена")
        for name in names:
            clean = posixpath.normpath(name)
            if name.startswith("/") or clean == ".." or clean.startswith("../"):
                fail(errors, f"небезопасный путь: {name}")
            parts = name.rstrip("/").split("/")
            if any(p == "__pycache__" or p.endswith(".pyc") for p in parts):
                fail(errors, f"служебный Python-файл: {name}")
            if any(p in FORBIDDEN for p in parts):
                fail(errors, f"запрещённый файл: {name}")

        missing_runtime = sorted(RUNTIME - name_set)
        if missing_runtime:
            fail(errors, "нет runtime-файлов: " + ", ".join(missing_runtime))
        for name in RUNTIME & name_set:
            if zf.getinfo(name).file_size == 0:
                fail(errors, f"пустой runtime-файл: {name}")
        for name in {"run.py", "pair_text.py", "attr_canon.py"} & name_set:
            try:
                compile(zf.read(name).decode("utf-8"), name, "exec")
            except (SyntaxError, UnicodeDecodeError) as exc:
                fail(errors, f"{name}: синтаксическая ошибка: {exc}")
        for name, want_type in (("brands.json", list), ("pri_by_cat.json", dict)):
            if name in name_set:
                try:
                    value = json.loads(zf.read(name))
                    if not isinstance(value, want_type):
                        fail(errors, f"{name}: ожидался {want_type.__name__}")
                except (ValueError, UnicodeDecodeError) as exc:
                    fail(errors, f"{name}: некорректный JSON: {exc}")

        model_dirs = sorted({n.split("/", 1)[0] for n in names
                             if n.startswith("model") and "/" in n})
        if not model_dirs:
            fail(errors, "нет каталогов model*")
        if args.expect_models and len(model_dirs) != args.expect_models:
            fail(errors, f"моделей {len(model_dirs)}, ожидалось {args.expect_models}")
        model_dtypes = {}
        for model in model_dirs:
            required = {f"{model}/config.json", f"{model}/model.safetensors"}
            missing = required - name_set
            if missing:
                fail(errors, f"{model}: нет {', '.join(sorted(missing))}")
            else:
                config_name = f"{model}/config.json"
                weights_name = f"{model}/model.safetensors"
                try:
                    config = json.loads(zf.read(config_name))
                    if not isinstance(config, dict) or not config:
                        fail(errors, f"{config_name}: пустой config")
                except (ValueError, UnicodeDecodeError) as exc:
                    fail(errors, f"{config_name}: некорректный JSON: {exc}")
                try:
                    info = zf.getinfo(weights_name)
                    with zf.open(weights_name) as fh:
                        _, dtypes = safetensors_header(fh, info.file_size)
                        model_dtypes[model] = dtypes
                        if args.require_fp16 and any(
                                dtype in {"F32", "F64"} for dtype in dtypes):
                            fail(errors, f"{weights_name}: некомпактные dtype {dtypes}")
                except (OSError, ValueError, KeyError, json.JSONDecodeError,
                        struct.error) as exc:
                    fail(errors, f"{weights_name}: некорректный safetensors: {exc}")
            tokenizers = {f"{model}/tokenizer.json", f"{model}/tokenizer_config.json"}
            if not (tokenizers & name_set):
                fail(errors, f"{model}: нет tokenizer")
            for tokenizer in tokenizers & name_set:
                if zf.getinfo(tokenizer).file_size == 0:
                    fail(errors, f"{tokenizer}: пустой tokenizer")

        for model in args.expect_ordered:
            if f"{model}/ORDERED" not in name_set:
                fail(errors, f"{model}: ожидался маркер ORDERED")

        for name in sorted(n for n in name_set
                           if n.endswith("/RRF_SCORER_WEIGHT")):
            model = name.split("/", 1)[0]
            if f"{model}/RRF_K" not in name_set:
                fail(errors, f"{name}: требует {model}/RRF_K")
            try:
                value = float(zf.read(name).decode("utf-8").strip())
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("ожидалось конечное положительное число")
            except (ValueError, UnicodeDecodeError) as exc:
                fail(errors, f"{name}: некорректный вес: {exc}")

        if "metadata.json" in name_set:
            try:
                metadata = json.loads(zf.read("metadata.json"))
                entry = metadata["entry_point"]
                if "run.py" not in entry:
                    fail(errors, "metadata.entry_point не запускает run.py")
                # Длина больше не фиксирована жёстко: обе модели — ModernBERT с
                # RoPE и max_position_embeddings 8192, а замер на стенде 1{k=9}
                # показал заметный выигрыш при 512. Проверка сохраняет смысл —
                # длина должна быть указана явно и быть из проверенного набора,
                # чтобы опечатка не уехала в сабмит.
                allowed = {"384", "512", "640"}
                found = None
                parts = entry.split()
                for i, part in enumerate(parts):
                    if part == "--max_len" and i + 1 < len(parts):
                        found = parts[i + 1]
                if found is None:
                    fail(errors, "metadata.entry_point не задаёт --max_len")
                elif found not in allowed:
                    fail(errors, f"--max_len {found} вне проверенного набора "
                                 f"{sorted(allowed)}")
                if not metadata.get("image"):
                    fail(errors, "metadata.image пуст")
            except (KeyError, ValueError, UnicodeDecodeError) as exc:
                fail(errors, f"некорректный metadata.json: {exc}")

        lookup = {"llm_key.npy", "llm_k.npy"}
        if bool(lookup & name_set) != lookup.issubset(name_set):
            fail(errors, "lookup неполный: llm_key.npy и llm_k.npy нужны вместе")
        if args.expect_lookup and not lookup.issubset(name_set):
            fail(errors, "ожидался exact LLM lookup")
        if args.expect_closure and "llm_pos_closure.npy" not in name_set:
            fail(errors, "ожидался llm_pos_closure.npy")
        if args.expect_textsig and "textsig_negative_u64.npy" not in name_set:
            fail(errors, "ожидался exact text-signature lookup")
        if "llm_pos_closure.npy" in name_set and not lookup.issubset(name_set):
            fail(errors, "closure разрешён только вместе с exact lookup")

        headers = {}
        for name in sorted(lookup | {
                "llm_pos_closure.npy", "textsig_negative_u64.npy"}):
            if name not in name_set:
                continue
            try:
                with zf.open(name) as fh:
                    headers[name] = npy_header(fh)
            except (OSError, ValueError, SyntaxError, struct.error) as exc:
                fail(errors, f"{name}: некорректный NPY header: {exc}")
        if lookup.issubset(headers):
            kd, ks, kf = headers["llm_key.npy"]
            vd, vs, vf = headers["llm_k.npy"]
            if kd not in ("<i8", "|i8") or vd not in ("|u1", "<u1"):
                fail(errors, f"неожиданные lookup dtype: key={kd}, k={vd}")
            if len(ks) != 1 or ks != vs or kf or vf:
                fail(errors, f"несогласованные lookup shapes: key={ks}, k={vs}")
        if "llm_pos_closure.npy" in headers:
            dtype, shape, fortran = headers["llm_pos_closure.npy"]
            if dtype not in ("<i8", "|i8") or len(shape) != 1 or fortran:
                fail(errors, f"некорректный closure header: {headers['llm_pos_closure.npy']}")
        if "textsig_negative_u64.npy" in headers:
            dtype, shape, fortran = headers["textsig_negative_u64.npy"]
            if dtype not in ("<u8", "|u8") or len(shape) != 2 \
                    or shape[1] != 4 or fortran:
                fail(errors, "некорректный text-signature header: "
                     f"{headers['textsig_negative_u64.npy']}")

        broken = zf.testzip()
        if broken:
            fail(errors, f"CRC не прошёл: {broken}")

    if errors:
        for message in errors:
            print(f"ОШИБКА: {message}", file=sys.stderr)
        return 1
    print(f"OK: {args.archive}")
    print(f"  SHA-256: {sha256(args.archive)}")
    print(f"  файлов: {len(names)}, моделей: {len(model_dirs)} ({', '.join(model_dirs)})")
    for model, dtypes in model_dtypes.items():
        print(f"  {model}: safetensors dtype={','.join(dtypes)}")
    if headers:
        for name, header in headers.items():
            print(f"  {name}: dtype={header[0]}, shape={header[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
