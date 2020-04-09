#Input the data set
smart_bin_data_set <- read.csv("data-5.csv",stringsAsFactors = FALSE)
#Converting the date-time format suitable consistent in R
smart_bin_5$timestamp <- as.POSIXct(smart_bin_5$timestamp,format="%m/%d/%Y %H:%M",tz="UTC")
#Sort the data w.r.t timestamp
smart_bin_5$timestamp<- sort(smart_bin_5$timestamp)
#Extracting the day of the week
smart_bin_5$Day_of_the_week<-weekdays.POSIXt(smart_bin_5$timestamp,abbreviate =FALSE)
#Extracting hours
smart_bin_5$hour <- format(smart_bin_5$timestamp,"%H")
#Extracting day hours
smart_bin_5$day_hour <- format(smart_bin_5$timestamp,"%d %H")
#formating the data
smart_bin_5$day_hour <-gsub(" ","-",smart_bin_5$day_hour)
# Creating function for time slots
time_slot_identifier <- function(x){if(x>=05 && x<=08){return("Early Morning")}
  else if(x>=09 && x<12){return("Late Morning")}
  else if (x>=12 && x <=15) {return("Early Afternoon")}
  else if (x>=16 && x <=17){return("Late Afternoon")}
  else if (x>=18 && x<=19){return("Early Evening")}
  else if (x >=20 && x<=23){return("Night")}
  else if (x>=00 && x<=02){return("Late Night")}
  else{return("Very Early Morning")}}
#Utlising the time slot identifier function to create time slots from the hours
smart_bin_5$Time_slots <-sapply(smart_bin_5$hour,time_slot_identifier)

#Plot created

bin_level_day_wise_plot <- ggplot(smart_bin_5,aes(x=smart_bin_5$Time_slots,y=smart_bin_5$binLevel,fill=smart_bin_5$Day_of_the_week))+geom_bar(position="dodge",stat="identity")+theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))+xlab("Time Slots")+ylab("Bin level percentage")
bin_level_day_wise_plot <-bin_level_day_wise_plot + labs(colour="Day Of The Week")
bin_level_day_wise_plot <-bin_level_day_wise_plot + labs(title="Hour-wise Bin level Plot")

#Creating Subset based on bin Level and day_hour attribute
smart_bin_subset <- smart_bin_5[,c("binLevel","day_hour","hour")]

#Train set
smart_bin_train <-smart_bin_subset[1:200,]

#Test set
smart_bin_test <- smart_bin_subset[201:235,]

#Creating time-series
smart_bin_train_ts <- ts(smart_bin_train$binLevel)

#Summary of the time series
summary(smart_bin_train_ts)

#Initial Plot of the time series
plot(smart_bin_train_ts)

#Fitting a line in the time series
abline(reg=lm(smart_bin_train_ts~time(smart_bin_train_ts)))

#Removing the seasonal variation and trend component. Making the time-series stationary
adf.test(diff(log(smart_bin_train_ts)), alternative="stationary", k=0)

#p-value comes to be =0.01 indicating its stationary series
#Testing the stationarity of the model
adf.test(diff(log(AirPassengers)), alternative="stationary", k=0)



#Model Building using Auto Arima Process

autoarima <- auto.arima(smart_bin_train_ts)
autoarima

#Coefficients:
#ar1      ar2     ar3      ma1     ma2     mean
#1.9361  -1.5179  0.3714  -1.5985  0.8056  39.4455
#s.e.  0.1133   0.1557  0.0912   0.0928  0.0840   1.4495
#sigma^2 estimated as 447.4:  log likelihood=-891.78
#AIC=1797.56   AICc=1798.14   BIC=1820.65

tsdiag(autoarima)

accuracy(autoarima)

#MAPE value=462.922

plot(autoarima$x, col="black")
lines(fitted(autoarima), col="red")

# to check whether residual series is stationary
resi_auto_arima <- smart_bin_train_ts -fitted(autoarima)
adf.test(resi_auto_arima,alternative = "stationary")

# Creating forecast
fcast_auto_arima <- predict(autoarima ,n.ahead=35)
mape_auto_arima <- accuracy(fcast_auto_arima$pred,smart_bin_test[,1])[5]
mape_auto_arima

#mape value=476.2853

auto_arima_pred <- c(fitted(autoarima),ts(fcast_auto_arima$pred))
smart_bin_subset_ts <- ts(smart_bin_subset$binLevel)
auto_arima_pred <- c(fitted(autoarima),ts(fcast_auto_arima$pred))
plot(smart_bin_subset_ts,col="black")
lines(auto_arima_pred,col="red")




