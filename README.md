#  TLS-MWP: A Tensor-based Long- and Short-range Convolution for Multiple Weather Prediction

Official source code for paper 《TLS-MWP: A Tensor-based Long- and Short-range Convolution for Multiple Weather Prediction》
### Overall Architecture of TLS-MWP
![image](https://github.com/xuguangning1218/TLS_MWP/blob/master/figure/model.png)

### Environment Installation
```
pip install -r requirements.txt
```  
### Reproducibility 
* Download the required Pressure-Level ERA5 (1000 hpa Relative Humidity) dataset through the official site in [here](<https://cds.climate.copernicus.eu/cdsapp/#!/dataset/reanalysis-era5-pressure-levels?tab=overview>  "here") and the required Single-Level ERA5 (2m Temperature, Surface Pressure, 10m Wind Speed) dataset through the official site in [here](<https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview> "here"). 
* Or using the well-prepare data and saved model in [here](<https://pan.baidu.com/s/1usug30aMdp0_RMoKbJaEWw?pwd=wn99>)

###  Source Files Description

```
-- data # data folder
	-- sample_test.npy # sample of the test dataset
	-- sample_train_validate.npy # sample of the train_validate dataset
-- data_loader # data loader folder
	-- era5.py # dataloader in train, validate, test for ERA5
	-- normalizer.py # data normalizer, including std, maxmin
-- figure # figure provider
	-- model.png # architecture of TLS-MWP model 
-- model # proposed model
	-- Encode2Decode.py # the frame-by-frame framework
	-- tls_convLSTM_cell.py # the ConvLSTM with TLS-Conv and the TLS-Conv
	-- Model.py # model handler of train, validate, test, etc.
requirements.txt # requirements package of the project
setting.config # model configure
Run.ipynb # jupyter visualized code for the model
```

###  Citation
If you think our work is helpful. Please kindly cite
```
@article{XU2022121,
title = {TLS-MWP: A Tensor-based Long- and Short-range Convolution for Multiple Weather Prediction},
journal = {IEEE Transactions on Circuits and Systems for Video Technology},
author = {Guangning Xu, Michael K. Ng, Yunming Ye, Xutao Li, Ge Song, Bowen Zhang, Zhichao Huang},
volume = {34},
number = {9},
pages = {8382-8397},
}
```
