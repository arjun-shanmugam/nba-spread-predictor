
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

 
/*----------------------------------------------------------*/
   /* [>   Cleaning games.csv (one row for every game)   <] */ 
/*----------------------------------------------------------*/
import delimited ${dir}/raw_data/games.csv, clear  // Load the data

keep game_id game_date_est pts_home pts_away  home_team_id visitor_team_id

generate point_differential = pts_home - pts_away  // to be used as our training labels
label variable point_differential "Home team's score - away team's score"  
drop pts_home pts_away

// there are some duplicate games because of an error by the dataset creator
sort game_id
quietly by game_id:  gen dup = cond(_N==1,0,_n)
drop if dup != 0


save ${dir}/intermediate_data/games.dta, replace

 
/*-----------------------------------------------------------------------------------*/
   /* [>   Cleaning games_details.csv (one row for every player in every game)    <] */ 
/*-----------------------------------------------------------------------------------*/
// players are not marked as "home" or "away" in games_details.csv, so we have to mark them ourselves
// start with home

// first, we convert games_details.csv to a .dta file so that we can merge
import delimited ${dir}/raw_data/games_details.csv, clear  // Load the data

// drop non-starters
drop if start_position == ""

// Replace F's with F1 and F2
replace start_position = "F1" if mod(_n - 1, 5) == 0  // the first forward of each team appears every 5 rows, starting from 1
replace start_position = "F2" if mod(_n -2, 5) == 0  // the second forward of each team appears every 5 rows, starting from 1

// Replace G's with G1 and G2
replace start_position = "G1" if mod(_n + 1, 5) == 0  // the first forward of each team appears every 5 rows, starting from 1
replace start_position = "G2" if mod(_n, 5) == 0  // the second forward of each team appears every 5 rows, starting from 1

save ${dir}/intermediate_data/games_details.dta, replace

 
/*--------------------------------------------------------------------------------*/
   /* [>   Merging games.dta with HOME TEAMS' GAME DATA from games_details.dta <] */ 
/*--------------------------------------------------------------------------------*/
// now, load games.dta
use ${dir}/intermediate_data/games.dta, clear

rename home_team_id team_id  // rename to match the games_details.dta file

merge 1:m game_id team_id using ${dir}/intermediate_data/games_details.dta

/*
    Result                      Number of obs
    -----------------------------------------
    Not matched                       313,538
        from master                        99  (_merge==1)  --> games for which we do not have individual player data
        from using                    313,439  (_merge==2)  --> data on the visiting team

    Matched                           312,672  (_merge==3)
    -----------------------------------------
*/
drop if _merge != 3
rename team_id home_team_id  // Reverse variable name change above




// drop unneeded variables
drop dup team_abbreviation team_city player_name nickname comment min fgm fga fg_pct fg3m fg3a fg3_pct ftm fta ft_pct oreb dreb reb ast stl blk to pf pts plus_minus _merge

// sort by game_id
sort game_id  // At this point, we have 5 rows for each game, corresponding to the five starters on the home team in each game

// now, combine each group of five rows into a single row
generate home_starting_c = player_id if start_position == "C" 
generate home_starting_f1 = player_id if start_position == "F1"
generate home_starting_f2 = player_id if start_position == "F2"
generate home_starting_g1 = player_id if start_position == "G1"
generate home_starting_g2 = player_id if start_position == "G2"

drop player_id start_position

collapse (firstnm) game_date_est home_team_id point_differential visitor_team_id home_starting_c home_starting_f1 home_starting_f2 home_starting_g1 home_starting_g2, by(game_id)



/*------------------------------------------------------------------------------*/
   /* [>   Merging games.dta with VISITORS' GAME DATA from games_details.dta <] */ 
/*------------------------------------------------------------------------------*/
// follow same procedure as above
rename visitor_team_id team_id

merge 1:m game_id team_id using ${dir}/intermediate_data/games_details.dta
/*
    Result                      Number of obs
    -----------------------------------------
    Not matched                       119,952
        from master                         0  (_merge==1)
        from using                    119,952  (_merge==2) --> data on the home teams

    Matched                           119,423  (_merge==3)
    -----------------------------------------
*/
drop if _merge != 3
rename team_id visitor_team_id  // Reverse variable name change above


drop team_abbreviation team_city player_name nickname comment min fgm fga fg_pct fg3m fg3a fg3_pct ftm fta ft_pct oreb dreb reb ast stl blk to pf pts plus_minus _merge

// sort by game_id
sort game_id  // At this point, we have 5 rows for each game, corresponding to the five starters on the home team in each game

// now, combine each group of five rows into a single row
generate visitor_starting_c = player_id if start_position == "C" 
generate visitor_starting_f1 = player_id if start_position == "F1"
generate visitor_starting_f2 = player_id if start_position == "F2"
generate visitor_starting_g1 = player_id if start_position == "G1"
generate visitor_starting_g2 = player_id if start_position == "G2"

drop player_id start_position

collapse (firstnm) game_date_est home_team_id point_differential visitor_team_id home_starting_c home_starting_f1 home_starting_f2 home_starting_g1 home_starting_g2 visitor_starting_c visitor_starting_f1 visitor_starting_f2 visitor_starting_g1 visitor_starting_g2, by(game_id)




label variable game_date_est "Date of game"
label variable home_team_id "ID of home team"
label variable point_differential "Home team's points - visiting team's points"
label variable visitor_team_id "ID of visiting team"
label variable home_starting_c "Home team's starting center"
label variable home_starting_f1 "One of home team's starting forwards"
label variable home_starting_f2 "Another of home team's starting forwards"
label variable home_starting_g1 "One of home team's starting guards"
label variable home_starting_g2 "Another of home team's starting guards"
label variable visitor_starting_c "Visiting team's starting center"
label variable visitor_starting_f1 "One of visiting team's starting forwards"
label variable visitor_starting_f2 "Another of visiting team's starting forwards"
label variable visitor_starting_g1 "One of visiting team's starting guards"
label variable visitor_starting_g2 "Another of visiting team's starting guards"

save ${dir}/intermediate_data/games.dta, replace

 
/*-----------------------------------------------------------------------*/
   /* [>   Merging games.dta with up-to-date team home team records THIS SECTION NOT FINISHSED  <] */ 
/*-----------------------------------------------------------------------*/
// first, load rankings.csv and clean it
import delimited ${dir}/raw_data/ranking.csv, clear 
drop league_id season_id conference team home_record road_record returntoplay 
rename standingsdate game_date_est  // so that we can merge on this variable

sort team_id game_date_est  // Remove scraping errors that created duplicates
quietly by team_id game_date_est:  gen dup = cond(_N==1,0,_n)
drop if dup != 0
drop dup

save ${dir}/intermediate_data/rankings.dta, replace

// reload games.dta
use ${dir}/intermediate_data/games.dta, clear

rename home_team_id team_id

// now, merge









