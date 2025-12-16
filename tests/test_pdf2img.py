from __future__ import annotations

import hashlib
from pathlib import Path

from pymupdf import Document
from tqdm import tqdm


def calculate_pdf_md5(pdf_path: Path) -> str:
    """计算PDF文件的MD5值"""
    md5_hash = hashlib.md5()
    with open(pdf_path, 'rb') as f:
        # 分块读取，避免大文件占用过多内存
        for chunk in iter(lambda: f.read(4096), b''):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def trans_pdf2png(pdf_path: Path, output_dir: Path):
    doc = Document(pdf_path)
    for page in enumerate(doc):
        page_index = page[0]
        page = page[1]
        dpis = [72, 144, 200]
        pix = None
        for dpi in dpis:
            pix = page.get_pixmap(dpi=dpi)
            if min(pix.width, pix.height) >= 1600:
                print(f"dpi: {dpi} is valid")
                break

        # 保存文件名为f"{输入pdf的md5值}_{page_index}.png"
        out_file = Path(output_dir) / f"{calculate_pdf_md5(pdf_path)}_page_{page_index}.png"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        pix.save(out_file.as_posix())


def trans_pdf2png_with_dpis(pdf_path: Path, output_dir: Path, dpi: int):
    doc = Document(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=dpi)
    for page_index, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        out_file = Path(output_dir) / f"{calculate_pdf_md5(pdf_path)}_page_{page_index}.png"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        pix.save(out_file.as_posix())


def run():
    input_dir = Path("E:/datasets/中广核图纸比对及信息提取/raw/sample")
    output_dir = Path("E:/datasets/中广核图纸比对及信息提取/raw/sample_png")
    for pdf_path in Path(input_dir).glob("*.pdf"):
        trans_pdf2png(pdf_path, output_dir)


def run2():
    input_dir = Path(r"E:\datasets\中广核图纸比对及信息提取\raw")
    output_dir = Path(r"E:\datasets\中广核图纸比对及信息提取\raw_png")
    for pdf_path in tqdm(list(Path(input_dir).glob("**/*.pdf"))):
        trans_pdf2png(pdf_path, output_dir)


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os

    data_dir = Path(r"E:\datasets\CGN\raw")
    idps = [72, 144, 200, 240]
    output_dir = Path(r"E:\datasets\CGN\processed")
    for idp in idps:
        output_dir_ = output_dir / f'idp_{idp}'
        output_dir_.mkdir(parents=True, exist_ok=True)

    pdf_paths = list(Path(data_dir).glob("**/*.pdf"))
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for pdf_path in pdf_paths:
            for idp in idps:
                output_dir_ = output_dir / f'idp_{idp}'
                futures.append(executor.submit(trans_pdf2png_with_dpis, pdf_path, output_dir_, idp))
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()
