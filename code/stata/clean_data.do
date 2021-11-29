
/**********************************************************************/
/*  NBA Spread Predictor

	
	Arjun Shanmugam, Andy Delworth, Josh Liu, Sid Somasi


    This file cleans and merges input data. */
/**********************************************************************/
	

clear all
set more off

/* [> Set your own path here <] */ 

if "`c(username)'" == "arjunshanmugam" {
global dir `"/Users/arjunshanmugam/Documents/School/Brown/Semester5/CSCI1470/final/nba-spread-predictor"'
}