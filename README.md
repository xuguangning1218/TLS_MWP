#  TLS-MWP: A Tensor-based Long- and Short-range Convolution for Multiple Weather Prediction

Official source code for paper 《TLS-MWP: A Tensor-based Long- and Short-range Convolution for Multiple Weather Prediction》
### Overall Architecture of LS-NTP
![image](https://github.com/xuguangning1218/TLS_MWP/blob/master/figure/model.png)

### Environment Installation
```
pip install -r requirements.txt
```  
### Data Preparation 
* Download the required Pressure-Level ERA5 (1000 hpa Relative Humidity) dataset through [here](<https://cds.climate.copernicus.eu/cdsapp/#!/dataset/reanalysis-era5-pressure-levels?tab=overview>  "here") and the required Single-Level ERA5 (2m Temperature, Surface Pressure, 10m Wind Speed) dataset through [here](<https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview> "here"). 


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
