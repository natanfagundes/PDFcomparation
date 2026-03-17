#!/usr/bin/env python3
"""
Dependências para instalar o Software:
  pip install PyMuPDF Pillow opencv-python-headless imagehash scikit-image numpy

Exemplos de uso:
  python  main.py documento.pdf imagem_busca.png
  python main.py documento.pdf imagem_busca.jpg --threshold 0.85 --verbose
  python main.py documento.pdf imagem_busca.png --method orb
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import Optional


try:
    import fitz
except ImportError:
    sys.exit("❌  PyMuPDF não encontrado.  Execute: pip install PyMuPDF ❌")

try:
    from PIL import Image
    import numpy as np
except ImportError:
    sys.exit("❌  Pillow/NumPy não encontrados.  Execute: pip install Pillow numpy ❌")

try:
    import cv2
except ImportError:
    sys.exit("❌  OpenCV não encontrado.  Execute: pip install opencv-python-headless ❌ ")

try:
    import imagehash
except ImportError:
    sys.exit("❌  imagehash não encontrado.  Execute: pip install imagehash ❌")

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    sys.exit("❌  scikit-image não encontrado.  Execute: pip install scikit-image ❌")


# ── Constantes ────────────────────────────────────────────────────────────────

# Resolução de renderização das páginas (DPI) — aumentar = mais lento, mais preciso
PDF_RENDER_DPI = 150

# Tamanho padrão para normalizar imagens antes da comparação SSIM
SSIM_RESIZE = (256, 256)

# Número mínimo de matches ORB para considerar "encontrado"
ORB_MIN_MATCHES = 15

# Hash size para pHash
PHASH_SIZE = 16



def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Converte imagem PIL (RGB) para array OpenCV (BGR)."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img: np.ndarray) -> Image.Image:
    """Converte array OpenCV (BGR) para imagem PIL (RGB)."""
    rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_image(path: str) -> Image.Image:
    """Carrega uma imagem de disco com tratamento de erro."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Não foi possível abrir a imagem '{path}': {e}")


# ── Extração de imagens do PDF ────────────────────────────────────────────────

def extract_images_from_pdf(pdf_path: str, verbose: bool = False):
    """
    Extrai todas as imagens embutidas no PDF usando PyMuPDF.

    Yields:
        (page_number: int, image_index: int, pil_image: Image.Image)
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Não foi possível abrir o PDF '{pdf_path}': {e}")

    total_pages = len(doc)
    total_images = 0

    if verbose:
        print(f" PDF carregado: {total_pages} página(s)")

    for page_num in range(total_pages):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_pil = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
                total_images += 1

                if verbose:
                    w, h = img_pil.size
                    print(f" Página {page_num + 1}, imagem #{img_index + 1}  ({w}×{h}px)")

                yield (page_num + 1, img_index + 1, img_pil)

            except Exception as e:
                if verbose:
                    print(f"  Página {page_num + 1}, imagem #{img_index + 1}: ignorada ({e})")
                continue

    doc.close()

    if total_images == 0:
        raise ValueError("O PDF não contém imagens embutidas detectáveis.")

    if verbose:
        print(f"✅  Total de imagens extraídas: {total_images}\n")


# ── Métodos de comparação ─────────────────────────────────────────────────────

def compare_phash(img_a: Image.Image, img_b: Image.Image) -> float:
    
    hash_a = imagehash.phash(img_a, hash_size=PHASH_SIZE)
    hash_b = imagehash.phash(img_b, hash_size=PHASH_SIZE)

    max_distance = PHASH_SIZE * PHASH_SIZE  # bits totais
    distance = hash_a - hash_b
    return 1.0 - (distance / max_distance)


def compare_orb(img_a: Image.Image, img_b: Image.Image) -> float:
   
    gray_a = cv2.cvtColor(pil_to_cv2(img_a), cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(pil_to_cv2(img_b), cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1000)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None or len(kp_a) < 5 or len(kp_b) < 5:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_a, des_b)
    matches = sorted(matches, key=lambda x: x.distance)

    # Filtra matches com distância aceitável (< 64 de 256 max)
    good_matches = [m for m in matches if m.distance < 64]

    if len(good_matches) < ORB_MIN_MATCHES:
        return 0.0

    # Normaliza pelo número de keypoints da menor imagem
    reference = min(len(kp_a), len(kp_b))
    return min(1.0, len(good_matches) / reference)


def compare_ssim(img_a: Image.Image, img_b: Image.Image) -> float:
    
    # Redimensiona ambas para o mesmo tamanho
    a_resized = img_a.resize(SSIM_RESIZE, Image.LANCZOS)
    b_resized = img_b.resize(SSIM_RESIZE, Image.LANCZOS)

    arr_a = np.array(a_resized.convert("L"))  # grayscale
    arr_b = np.array(b_resized.convert("L"))

    score, _ = ssim(arr_a, arr_b, full=True)
    return max(0.0, float(score))


def compare_images(
    query: Image.Image,
    candidate: Image.Image,
    method: str = "phash",
) -> float:
    """
        Score de similaridade entre 0.0 e 1.0
    """
    method = method.lower()

    if method == "phash":
        return compare_phash(query, candidate)
    elif method == "orb":
        return compare_orb(query, candidate)
    elif method == "ssim":
        return compare_ssim(query, candidate)
    elif method == "all":
        s_phash = compare_phash(query, candidate)
        s_orb   = compare_orb(query, candidate)
        s_ssim  = compare_ssim(query, candidate)
        return (s_phash * 0.4) + (s_orb * 0.35) + (s_ssim * 0.25)
    else:
        raise ValueError(f"Método desconhecido: '{method}'. Use: phash, orb, ssim, all")


# ── Lógica principal de busca ─────────────────────────────────────────────────

def find_image_in_pdf(
    pdf_path: str,
    query_path: str,
    method: str = "phash",
    threshold: float = 0.85,
    verbose: bool = False,
) -> dict:
    
    start = time.time()

    # Carrega a imagem de consulta
    query_img = load_image(query_path)
    qw, qh = query_img.size

    if verbose:
        print(f"  Imagem de busca: {query_path}  ({qw}×{qh}px)")
        print(f"   Método: {method.upper()}  |  Threshold: {threshold:.0%}\n")

    matches = []
    best_score = 0.0

    for page_num, img_idx, pdf_img in extract_images_from_pdf(pdf_path, verbose=verbose):
        score = compare_images(query_img, pdf_img, method=method)

        if verbose:
            bar = "█" * int(score * 20)
            print(f"   Pág {page_num:>3}, img #{img_idx}: {score:.1%}  [{bar:<20}]")

        if score > best_score:
            best_score = score

        if score >= threshold:
            matches.append({
                "page": page_num,
                "image_index": img_idx,
                "similarity": score,
            })

    elapsed = time.time() - start

    return {
        "found": len(matches) > 0,
        "matches": sorted(matches, key=lambda x: x["similarity"], reverse=True),
        "best_score": best_score,
        "elapsed_sec": elapsed,
    }


# ── Interface de linha de comando ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf_image_finder",
        description="Verifica se uma imagem aparece em um PDF e indica as páginas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python main.py relatorio.pdf logo.png
  python main.py relatorio.pdf foto.jpg --threshold 0.80 --verbose
  python main.py relatorio.pdf imagem.png --method orb --threshold 0.70
  python main.py relatorio.pdf imagem.png --method all  --verbose
        """,
    )

    parser.add_argument("pdf",   help="Caminho do arquivo PDF")
    parser.add_argument("image", help="Caminho da imagem de busca")

    parser.add_argument(
        "--method", "-m",
        choices=["phash", "orb", "ssim", "all"],
        default="phash",
        help="Método de comparação (padrão: phash)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.85,
        metavar="0-1",
        help="Limiar de similaridade 0–1 (padrão: 0.85)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Exibe detalhes de cada imagem analisada",
    )

    return parser


def print_result(result: dict, threshold: float) -> None:
    """Imprime o resultado de forma legível no terminal."""
    sep = "─" * 55

    print(f"\n{sep}")

    if result["found"]:
        print(f"IMAGEM ENCONTRADA NO PDF! ")
        print(f"{sep}")
        pages = sorted(set(m["page"] for m in result["matches"]))
        print(f"Páginas: {', '.join(str(p) for p in pages)}")
        print()
        print(f"  {'Página':>6}  {'Img #':>5}  {'Similaridade':>14}")
        print(f"  {'──────':>6}  {'─────':>5}  {'──────────────':>14}")
        for m in result["matches"]:
            bar = "●" * int(m["similarity"] * 10)
            print(
                f"  {m['page']:>6}  {m['image_index']:>5}  "
                f"{m['similarity']:>12.1%}  {bar}"
            )
    else:
        print(f"❌  IMAGEM NÃO ENCONTRADA NO PDF")
        print(f"{sep}")
        print(f"   Maior similaridade detectada: {result['best_score']:.1%}")
        print(f"   Threshold configurado:        {threshold:.1%}")
        if result["best_score"] >= threshold * 0.8:
            print(
                f"\n   💡  Dica: a melhor correspondência ficou próxima do limiar.\n"
                f"       Tente --threshold {result['best_score'] * 0.95:.2f} para incluí-la."
            )

    print(f"\n Tempo de execução: {result['elapsed_sec']:.2f}s")
    print(sep)


def main() -> int:
    parser = build_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    if not (0.0 <= args.threshold <= 1.0):
        parser.error("--threshold deve estar entre 0.0 e 1.0")

    try:
        result = find_image_in_pdf(
            pdf_path=args.pdf,
            query_path=args.image,
            method=args.method,
            threshold=args.threshold,
            verbose=args.verbose,
        )
        print_result(result, args.threshold)
        return 0 if result["found"] else 1

    except FileNotFoundError as e:
        print(f"\n Arquivo não encontrado: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"\nErro de dados: {e}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
