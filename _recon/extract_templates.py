import json, hashlib, pathlib

src = pathlib.Path(r"D:\AB_Text\_recon")
out = pathlib.Path(r"D:\AB_Text\configs\templates")
out.mkdir(parents=True, exist_ok=True)
for name, f in [
    ("qwen3-4b.jinja", "qwen3_tokenizer_config.json"),
    ("deepseek-r1-distill-qwen-1.5b.jinja", "deepseek_tokenizer_config.json"),
]:
    cfg = json.loads((src / f).read_text(encoding="utf-8"))
    tpl = cfg["chat_template"]
    p = out / name
    p.write_text(tpl, encoding="utf-8")
    print(name, "sha256=", hashlib.sha256(tpl.encode("utf-8")).hexdigest(), "chars=", len(tpl))
