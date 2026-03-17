# Cria um ambiente virtual na pasta do projeto
py -m venv venv

# Ativa o ambiente
venv\Scripts\activate

# Instala tudo dentro do ambiente
pip install PyMuPDF Pillow opencv-python-headless imagehash scikit-image numpy

# Exemplo
python main.py atarde.pdf imagem2.jpeg
