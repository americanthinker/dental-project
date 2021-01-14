# Dental Patient Churn Prediction    

<p align="center">
  <img align="center" src="/images/churn.png" width="600" title="Predictive Analytics">
</p>  

I partnered with the CEO of a Private Equity backed dental practice with the understanding that:  
  <p align='center'>
  Business  +  Data Science  =  Better Business
  </p>
  <br>
  To that end I agreed to the following deliverables for his practice:  
  
  1. Build a patient churn prediction model with the following specs:  
        - Predictions based solely on patient characteristics (i.e. not based on time to patient's next appointment)  
        - Predictions are deployed within staff daily worksflow, thereby allowing them to take a proactive action to prevent churn. 
    
  2. Build a prioritized patient contact list (customer segmentation) for patients who do churn:   
        - List must be deployed into staff daily workflow to allow them to take action on contacting inactive patients in a prioritized fashion.  
  
## The Data
I had access to the entire practice's data which was housed in a mySQL database, comprising over 300 tables.  Only a handful were relevant to this particular project, 
among which were the following tables:   

<p float="left">
  <img src="/images/mysql.png" width="150" />
  <img src="/images/datatable.png" width="600" /> 
</p>

## Why it Matters: The Business of Churn 
Patient/customer churn is one of the primary drivers of revenue loss at any business.  However, in order to take meaningful action
to prevent or mitigate churn a business must first decide how it wants to define "churn".  The metrics used to define churn will be different across 
industries. For this project, churn is defined as a patient who does not have an appointment in the future and has not been seen at the practice in the last 400 days.  

The breakdown in numbers below shows how costly churn can be for this specific dental practice: 

<p align="center">
  <img align="center" src="/images/churn_biz2.png" width="600">
</p>


