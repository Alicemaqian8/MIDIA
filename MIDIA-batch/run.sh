#!/bin/bash
#percentages=('5%' '10%' '15%' '20%' '25%' '30%' '35%' '40%' '45%' '50%' '55%' '60%' '65%' '70%' '75%' '80%')
percentages=('5%' '10%' '15%' '20%' '25%' '30%' '35%' '40%' '45%' '50%')

for percentage in ${percentages[*]}
do
	echo $percentage
	cp ../dataset/AirQuality/various-miss-ratios/$percentage/test_data.txt ../dataset/AirQuality/test_data.txt
	cp ../dataset/AirQuality/various-miss-ratios/$percentage/train_data.txt ../dataset/AirQuality/train_data.txt
	cp ../dataset/AirQuality/various-miss-ratios/$percentage/univariate/test_indicator.txt ../dataset/AirQuality/test_indicator.txt
	echo $percentage >> univariate.txt
	#/usr/local/bin/python3.7 main.py >> result.txt
	E:/Programs/Python/Python3.7.2/python main.py >> univariate.txt
	#python main.py >> result.txt
done


for percentage in ${percentages[*]}
do
	echo $percentage
	cp ../dataset/AirQuality/various-miss-ratios/$percentage/test_data.txt ../dataset/AirQuality/test_data.txt
	cp ../dataset/AirQuality/various-miss-ratios/$percentage/train_data.txt ../dataset/AirQuality/train_data.txt
	cp ../dataset/AirQuality/various-miss-ratios/$percentage/monotone/test_indicator.txt ../dataset/AirQuality/test_indicator.txt
	echo $percentage >> monotone.txt
	#/usr/local/bin/python3.7 main.py >> result.txt
	E:/Programs/Python/Python3.7.2/python main.py >> monotone.txt
	#python main.py >> result.txt
done

for percentage in ${percentages[*]}
do
	echo $percentage
	cp ../dataset/AirQuality/various-miss-ratios/$percentage/test_data.txt ../dataset/AirQuality/test_data.txt
	cp ../dataset/AirQuality/various-miss-ratios/$percentage/train_data.txt ../dataset/AirQuality/train_data.txt
	cp ../dataset/AirQuality/various-miss-ratios/$percentage/general/test_indicator.txt ../dataset/AirQuality/test_indicator.txt
	echo $percentage >> general.txt
	#/usr/local/bin/python3.7 main.py >> result.txt
	E:/Programs/Python/Python3.7.2/python main.py >> general.txt
	#python main.py >> result.txt
done

