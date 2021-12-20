import torchvision.datasets as datasets
import os

os.environ['KAGGLE_USERNAME'] = 'khoaxxx238'
os.environ['KAGGLE_KEY'] = 'f4a2fee3d73eb41495880eec2a32d197'

os.system('pip install kaggle')
os.system('kaggle datasets download veeralakrishna/200-bird-species-with-11788-images --unzip')

ROOT = 'data'

datasets.utils.extract_archive('CUB_200_2011.tgz', ROOT)
