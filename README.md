# DeepUTI
This repository contains code for the experiments shown in the paper "A clinical microscopy dataset to develop a deep learning diagnostic test for urinary tract infection" [Nat. Sci. Data paper link](https://www.nature.com/articles/s41597-024-02975-0)

## Citation 
Please cite as follows:

Liou, N., De, T., Urbanski, A., Chieng, C., Kong, Q., David, A.L., Khasriya, R., Yakimovich, A., Horsley, H.: A clinical microscopy dataset to develop a deep learning diagnostic test for urinary tract infection. Sci Data. 11, 155 (2024). https://doi.org/10.1038/s41597-024-02975-0.

```
@ARTICLE{Liou24-deeputi,
  title     = "A clinical microscopy dataset to develop a deep learning
               diagnostic test for urinary tract infection",
  author    = "Liou, Natasha and De, Trina and Urbanski, Adrian and Chieng,
               Catherine and Kong, Qingyang and David, Anna L and Khasriya,
               Rajvinder and Yakimovich, Artur and Horsley, Harry",
  journal   = "Sci. Data",
  publisher = "Springer Science and Business Media LLC",
  volume    =  11,
  number    =  1,
  pages     = "155",
  month     =  feb,
  year      =  2024,
  copyright = "https://creativecommons.org/licenses/by/4.0"
}
```

## Abstract
Urinary tract infection (UtI) is a common disorder. Its diagnosis can be made by microscopic examination of voided urine for markers of infection. This manual technique is technically difficult, time-consuming and prone to inter-observer errors. the application of computer vision to this domain has been slow due to the lack of a clinical image dataset from UtI patients. We present an open dataset containing 300 images and 3,562 manually annotated urinary cells labelled into seven classes of clinically significant cell types. It is an enriched dataset acquired from the unstained and untreated urine of patients with symptomatic UtI using a simple imaging system. We demonstrate that this dataset can be used to train a Patch U-Net, a novel deep learning architecture with a random patch generator to recognise urinary cells. Our hope is, with this dataset, UTI diagnosis will be made possible in nearly all clinical settings by using a simple imaging system which leverages advanced machine learning techniques.

## Installation
Clone the repository and set up environment for HZDR Hemera users:
```
    git clone https://github.com/casus/UMOD.git
    cd UMOD
    module load cuda/11.6
    conda env create -n deeputi -f environment.yml
    conda activate deeputi
    pip install -e .
    ln -s /bigdata/casus/MLID/DeepUTI/ds1 data
```
For others:
```
    git clone https://github.com/casus/UMOD.git
    cd UMOD
    conda env create -n deeputi -f environment.yml
    conda activate deeputi
    pip install -e .
```   

## Troubleshooting
For HZDR Hemera users:
```
RuntimeError: Internal: libdevice not found at ./libdevice.10.bc
```

Solved by
```
ln -s /trinity/shared/pkg/devel/cuda/11.6 ./cuda_sdk_lib
```

## Usage
Please use the scripts under ```scripts/``` and notebooks under ```notebooks/``` and ```train_model.py``` in combination with an appropriate config file from ```configs/``` in the ```.json``` format to run the code. Please change the config with appropriate data, output and model weight paths.


## License
This repository is shared under the MIT License.
