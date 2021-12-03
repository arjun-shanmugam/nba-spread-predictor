
/**********************************************************************/
/*  NBA Spread Predictor

	
	Arjun Shanmugam, Andy Delworth, Josh Liu, Sid Somasi


    This file cleans and merges input data. */
/**********************************************************************/
clear all
set more off

/* [> Set your own path here if you want to run the file<] */ 
if "`c(username)'" == "arjunshanmugam" {
global dir `"/Users/arjunshanmugam/Documents/School/Brown/Semester5/CSCI1470/final/nba-spread-predictor"'
}




/*----------------------------------------------------------*/
   /* [>   Cleaning games.csv (one row for every game)   <] */ 
/*----------------------------------------------------------*/
import delimited ${dir}/raw_data/games.csv, clear  // load the data

keep game_id game_date_est pts_home pts_away home_team_id visitor_team_id  // delete unneeded vars

generate point_differential = pts_home - pts_away  // to be used as our training labels
label variable point_differential "Home team's score - away team's score"  
drop pts_home pts_away

// there are some game_id's that appear twice because of a scraping error by the dataset creator
sort game_id
quietly by game_id:  gen dup = cond(_N==1,0,_n)
drop if dup != 0  // drop them

save ${dir}/intermediate_data/games.dta, replace  // save the cleaned csv dile as a .dta




/*-----------------------------------------------------------------------------------*/
   /* [>   Cleaning games_details.csv (one row for every player in every game)    <] */ 
/*-----------------------------------------------------------------------------------*/
import delimited ${dir}/raw_data/games_details.csv, clear  // load the data

drop if start_position == ""  // drop non-starters

/* [> 
We need to sort by player_id and game_date_est so that we know a players avg. performance over prev. ten games going into each game.
The game date variable, game_date_est, only exists in games.dta.
We also need to mark each player as "home" or "away"; home_team_id only exists in games.dta.
Thus, we execute a merge to match each row in games_details.csv (one row for every player in every game) to a date.
 <] */ 
merge m:1 game_id using "${dir}/intermediate_data/games.dta"
/* 
   Result                      Number of obs
   -----------------------------------------
   Not matched                         1,586
       from master                       529  (_merge==1) -> rows in games_details.csv that do not correspond to game_id's in games.dta
       from using                      1,057  (_merge==2) -> rows in games.dta that do not have corresponding player data in games_details.csv

   Matched                           238,846  (_merge==3)
   -----------------------------------------
*/
drop if _merge != 3

gen home = 0
replace home = 1 if team_id == home_team_id  // set home indicator to 1 if player's team_id = home_team_id from games.dta

// drop all variables except for game date
drop home_team_id visitor_team_id point_differential dup

// sort by player_id and game date
sort player_id game_date_est

// drop observations that are missing game_date_est
drop if game_date_est == ""

/* [> for each row, calculate mean of the below stats over the past ten games <] */ 
local statvars fgm fga fg_pct fg3m fg3a fg3_pct ftm fta ft_pct oreb dreb reb ast stl blk to pf pts plus_minus
foreach var of varlist `statvars' {
#delimit ;
by player_id: gen mean_`var' =
	(`var'[_n - 10] + `var'[_n - 9] + `var'[_n - 8] + `var'[_n - 7] + `var'[_n - 6] +
	 `var'[_n - 5] + `var'[_n - 4] + `var'[_n - 3] + `var'[_n - 2] + `var'[_n - 1]) / 10 ;
#delimit cr
}

drop `statvars' _merge // drop original (non averaged) variables and the merge indicator

save ${dir}/intermediate_data/games_details.dta, replace  // save the dataset


 

/*-------------------------------------------------------*/
   /* [>   Merging games_details.dta with games.dta   <] */ 
/*-------------------------------------------------------*/
/* [> we will merge vectors of player data with each game, position by position <] */ 
/* [> order: home centers, visiting centers, home forwards, visiting forwards, home guards, away guards <] */ 

// store the variables giving the means of each stat over previous ten games 
#delimit ;
local mean_statvars mean_fgm mean_fga mean_fg_pct  
mean_fg3m mean_fg3a mean_fg3_pct mean_ftm
mean_fta mean_ft_pct mean_oreb mean_dreb
mean_reb mean_ast mean_stl mean_blk
mean_to mean_pf mean_pts mean_plus_minus;
#delimit cr


/* [> HOME CENTERS <] */ 
keep if start_position == "C" // drop if not a center
keep if home == 1  // drop if not playing at home
foreach var of varlist `mean_statvars' {  // add home_c_ to front of varnames to indicate that these stats are for home centers
rename `var' home_c_`var'
}

// there are 8 duplicate games (among home centers) because of an error by the original dataset's creator
bysort game_id:  gen dup = cond(_N==1,0,_n)
drop if dup != 0  // drop them
drop dup  // drop the tool variable we used to identify duplicates

// keep only game_id and the statistics variables
keep game_id home_c_mean_fgm home_c_mean_fga home_c_mean_fg_pct home_c_mean_fg3m home_c_mean_fg3a home_c_mean_fg3_pct home_c_mean_ftm home_c_mean_fta home_c_mean_ft_pct home_c_mean_oreb home_c_mean_dreb home_c_mean_reb home_c_mean_ast home_c_mean_stl home_c_mean_blk home_c_mean_to home_c_mean_pf home_c_mean_pts home_c_mean_plus_minus


// merge each row (a single home center's stats in a single game) with a single game in games.dta
merge 1:1 game_id using "${dir}/intermediate_data/games.dta"
/*
    Result                      Number of obs
    -----------------------------------------
    Not matched                         1,063
        from master                         0  (_merge==1)
        from using                      1,063  (_merge==2) -> this is likely b/c of a scraping error by author

    Matched                            23,873  (_merge==3)
    -----------------------------------------
*/
drop if _merge != 3
drop _merge

save ${dir}/intermediate_data/games.dta, replace  // save the dataset


/* [> VISITING CENTERS <] */ 
use ${dir}/intermediate_data/games_details.dta, clear  // reload the games_details.dta file (with all players, not just home centers)
keep if start_position == "C" // drop if not a center
keep if home == 0  // drop if not a visitor
foreach var of varlist `mean_statvars' {  // add away_c_ to front of varnames to indicate that these stats are for visiting centers
rename `var' away_c_`var'
}

// there are 4 duplicate games (among visiting centers) because of an error by the dataset creator
bysort game_id:  gen dup = cond(_N==1,0,_n)
drop if dup != 0  // drop them
drop dup  // drop the tool variable we used to identify duplicates

// keep only game_id and the statistics variables 
keep game_id away_c_mean_fgm away_c_mean_fga away_c_mean_fg_pct away_c_mean_fg3m away_c_mean_fg3a away_c_mean_fg3_pct away_c_mean_ftm away_c_mean_fta away_c_mean_ft_pct away_c_mean_oreb away_c_mean_dreb away_c_mean_reb away_c_mean_ast away_c_mean_stl away_c_mean_blk away_c_mean_to away_c_mean_pf away_c_mean_pts away_c_mean_plus_minus

// merge each row (a single visiting center's stats in a single game) with a single game in games.dta 
merge 1:1 game_id using "${dir}/intermediate_data/games.dta"
/*
    Result                      Number of obs
    -----------------------------------------
    Not matched                             8  -> cleaning error; reach out to dataset author about script used to webscrape
        from master                         6  (_merge==1)
        from using                          2  (_merge==2)

    Matched                            23,871  (_merge==3)
    -----------------------------------------
*/
drop if _merge != 3
drop _merge

save ${dir}/intermediate_data/games.dta, replace


/* [> HOME FORWARDS <] */ 
/*
Concatenating forwards' statistics is more complicated because there are two home forwards and two visitor forwards in each game.
They are not uniquely identified; i.e., each is marked with an "F."
The forwards data is organized in "long" format: each forward has his own row so that there are four rows per game instead of
	just one.
We will use the reshape command to put the forwards data in "wide" format: a single row for each game that concatenates the stats
	of all forwards.
Start with home forwards.
*/
use ${dir}/intermediate_data/games_details.dta, clear  // reload the games_details.dta file
keep if start_position == "F"  // drop if not a forward
keep if home == 1  // drop if not playing at home
foreach var of varlist `mean_statvars' {  // add home_f_ to front of varnames to indicate that these stats are for home forwards
rename `var' home_f_`var'
}

// keep only game_id, player_id, and the statistics variables 
keep game_id player_id home_f_mean_fgm home_f_mean_fga home_f_mean_fg_pct home_f_mean_fg3m home_f_mean_fg3a home_f_mean_fg3_pct home_f_mean_ftm home_f_mean_fta home_f_mean_ft_pct home_f_mean_oreb home_f_mean_dreb home_f_mean_reb home_f_mean_ast home_f_mean_stl home_f_mean_blk home_f_mean_to home_f_mean_pf home_f_mean_pts home_f_mean_plus_minus

// there are some games which mistakenly list the same forward as playing in the same game twice; delete them
sort game_id player_id
quietly by game_id player_id:  gen dup = cond(_N==1,0,_n)
tab dup
drop if dup != 0  // drop them
drop dup  // drop the tool variable we used to identify duplicates

// now, we need to reshape the data. right now, every game has two rows (one for the stats of each home forward--remember, we dropped visitors)
// we want to turn these two rows into one row, with both forwards' stats concatenated

// in order to reshape wide, we need to assign each home forward in each game the number 1 or 2
by game_id: replace player_id = 1 if _n == 1
by game_id: replace player_id = 2 if _n == 2

// time to reshape
local forwardstats home_f_mean_fgm home_f_mean_fga home_f_mean_fg_pct home_f_mean_fg3m home_f_mean_fg3a home_f_mean_fg3_pct home_f_mean_ftm home_f_mean_fta home_f_mean_ft_pct home_f_mean_oreb home_f_mean_dreb home_f_mean_reb home_f_mean_ast home_f_mean_stl home_f_mean_blk home_f_mean_to home_f_mean_pf home_f_mean_pts home_f_mean_plus_minus
reshape wide `forwardstats', i(game_id) j(player_id)

// execute merge according to game id; now that both home forwards in each game have been combined into one row, merge is 1:1
merge 1:1 game_id using "${dir}/intermediate_data/games.dta"
/*
    Result                      Number of obs
    -----------------------------------------
    Not matched                            19
        from master                        19  (_merge==1) --> forwards in the games_details.dta dataset who do not have a matching game in games.dta
        from using                          0  (_merge==2)

    Matched                            47,753  (_merge==3)
    -----------------------------------------
*/
// at this point, there are two rows for every game: one for each of the home forwards starting in that game
drop if _merge != 3
drop _merge

save ${dir}/intermediate_data/games.dta, replace

/* [> VISITING FORWARDS <] */ 
/* See discussion of why cleaning forwards' data is difficult above. */
use ${dir}/intermediate_data/games_details.dta, clear  // reload the games_details.dta file
keep if start_position == "F"  // drop if not a forward
keep if home == 0  // drop if not visiting
foreach var of varlist `mean_statvars' {  // add away_f_ to front of varnames to indicate that these stats are for home forwards
rename `var' away_f_`var'
}

// keep only game_id, player_id, and the statistics variables 
keep game_id player_id away_f_mean_fgm away_f_mean_fga away_f_mean_fg_pct away_f_mean_fg3m away_f_mean_fg3a away_f_mean_fg3_pct away_f_mean_ftm away_f_mean_fta away_f_mean_ft_pct away_f_mean_oreb away_f_mean_dreb away_f_mean_reb away_f_mean_ast away_f_mean_stl away_f_mean_blk away_f_mean_to away_f_mean_pf away_f_mean_pts away_f_mean_plus_minus

// there are some games which mistakenly list the same forward as playing in the same game twice; delete them
bysort game_id player_id:  gen dup = cond(_N==1,0,_n)
tab dup
drop if dup != 0  // drop them
drop dup  // drop the tool variable we used to identify duplicates

// now, we need to reshape the data. right now, every game has two rows (one for the stats of each forward)
// we want to turn these two rows into one row, with both forwards' stats concatenated

// in order to reshape wide, we need to assign each awway forward in each game the number 1 or 2
by game_id: replace player_id = 1 if _n == 1
by game_id: replace player_id = 2 if _n == 2

// time to reshape
local forwardstats away_f_mean_fgm away_f_mean_fga away_f_mean_fg_pct away_f_mean_fg3m away_f_mean_fg3a away_f_mean_fg3_pct away_f_mean_ftm away_f_mean_fta away_f_mean_ft_pct away_f_mean_oreb away_f_mean_dreb away_f_mean_reb away_f_mean_ast away_f_mean_stl away_f_mean_blk away_f_mean_to away_f_mean_pf away_f_mean_pts away_f_mean_plus_minus
reshape wide `forwardstats', i(game_id) j(player_id)

// execute merge according to game id; now that both away forwards in each game have been combined into one row, merge is 1:1
merge 1:1 game_id using "${dir}/intermediate_data/games.dta"
/*
    Result                      Number of obs
    -----------------------------------------
    Not matched                             8
        from master                         7  (_merge==1)
        from using                          1  (_merge==2)

    Matched                            23,870  (_merge==3)
    -----------------------------------------
*/
// At this point, there are two rows for every game: one for each of the home forwards starting in that game
drop if _merge != 3
drop _merge

save ${dir}/intermediate_data/games.dta, replace


/* [> HOME GUARDS <] */ 
/*
Concatenating guards' statistics is more complicated because there are two home guards and two visitor guards in each game.
They are not uniquely identified; i.e., each is marked with an "G."
The guards data is organized in "long" format: each guard has his own row so that there are four rows per game instead of
	just one.
We will use the reshape command to put the guards data in "wide" format: a single row for each game that concatenates the stats
	of all the guards.
Start with home forwards.
*/
use ${dir}/intermediate_data/games_details.dta, clear  // reload the games_details.dta file
keep if start_position == "G"  // drop if not a guard
keep if home == 1 // drop if not home
foreach var of varlist `mean_statvars' {  // add home_g_ to front of varnames to indicate that these stats are for home guards
rename `var' home_g_`var'
}

// keep only game_id, player_id, and statistics variables 
keep game_id player_id home_g_mean_fgm home_g_mean_fga home_g_mean_fg_pct home_g_mean_fg3m home_g_mean_fg3a home_g_mean_fg3_pct home_g_mean_ftm home_g_mean_fta home_g_mean_ft_pct home_g_mean_oreb home_g_mean_dreb home_g_mean_reb home_g_mean_ast home_g_mean_stl home_g_mean_blk home_g_mean_to home_g_mean_pf home_g_mean_pts home_g_mean_plus_minus

// now, we need to reshape the data. right now, every game has two rows (one for the stats of each guard)
// we want to turn these two rows into one row, with both guards' stats concatenated

// there are some games which mistakenly list the same guard as playing in the same game twice; delete them
bysort game_id player_id:  gen dup = cond(_N==1,0,_n)
tab dup
drop if dup != 0  // drop them
drop dup  // drop the tool variable we used to identify duplicates

// in order to reshape wide, we need to assign each home guard in each game the number 1 or 2
by game_id: replace player_id = 1 if _n == 1
by game_id: replace player_id = 2 if _n == 2

local guardstats home_g_mean_fgm home_g_mean_fga home_g_mean_fg_pct home_g_mean_fg3m home_g_mean_fg3a home_g_mean_fg3_pct home_g_mean_ftm home_g_mean_fta home_g_mean_ft_pct home_g_mean_oreb home_g_mean_dreb home_g_mean_reb home_g_mean_ast home_g_mean_stl home_g_mean_blk home_g_mean_to home_g_mean_pf home_g_mean_pts home_g_mean_plus_minus
reshape wide `guardstats', i(game_id) j(player_id)

// execute merge according to game id; now that both home guards in each game have been combined into one row, merge is 1:1
merge 1:1 game_id using "${dir}/intermediate_data/games.dta"
/*

    Result                      Number of obs
    -----------------------------------------
    Not matched                            13
        from master                         9  (_merge==1)
        from using                          4  (_merge==2)

    Matched                            23,866  (_merge==3)
    -----------------------------------------
*/
drop if _merge != 3
drop _merge

save ${dir}/intermediate_data/games.dta, replace


/* [> AWAY GUARDS <] */
/* See discussion of why cleaning forwards' data is difficult above. */
use ${dir}/intermediate_data/games_details.dta, clear  // reload the games_details.dta file
keep if start_position == "G"  // drop if not a guard
keep if home == 0 // drop if not visiting
foreach var of varlist `mean_statvars' {  // add away_g_ to front of varnames to indicate that these stats are for away guards
rename `var' away_g_`var'
}

// keep game_id, player_id, and stats
keep game_id player_id away_g_mean_fgm away_g_mean_fga away_g_mean_fg_pct away_g_mean_fg3m away_g_mean_fg3a away_g_mean_fg3_pct away_g_mean_ftm away_g_mean_fta away_g_mean_ft_pct away_g_mean_oreb away_g_mean_dreb away_g_mean_reb away_g_mean_ast away_g_mean_stl away_g_mean_blk away_g_mean_to away_g_mean_pf away_g_mean_pts away_g_mean_plus_minus

// there are some games which mistakenly list the same guard as playing in the same game twice; delete them
bysort game_id player_id:  gen dup = cond(_N==1,0,_n)
tab dup
drop if dup != 0  // drop them
drop dup  // drop the tool variable we used to identify duplicates

// now, we need to reshape the data. right now, every game has two rows (one for the stats of each guard)
// we want to turn these two rows into one row, with both forwards stats concatenated

// in order to reshape wide, we need to assign each away guard in each game the number 1 or 2
by game_id: replace player_id = 1 if _n == 1
by game_id: replace player_id = 2 if _n == 2

local guardstats away_g_mean_fgm away_g_mean_fga away_g_mean_fg_pct away_g_mean_fg3m away_g_mean_fg3a away_g_mean_fg3_pct away_g_mean_ftm away_g_mean_fta away_g_mean_ft_pct away_g_mean_oreb away_g_mean_dreb away_g_mean_reb away_g_mean_ast away_g_mean_stl away_g_mean_blk away_g_mean_to away_g_mean_pf away_g_mean_pts away_g_mean_plus_minus
reshape wide `guardstats', i(game_id) j(player_id)

// execute merge according to game id; now that both away guards in each game have been combined into one row, merge is 1:1
merge 1:1 game_id using "${dir}/intermediate_data/games.dta"
/*
    Result                      Number of obs
    -----------------------------------------
    Not matched                            13
        from master                        13  (_merge==1)
        from using                          0  (_merge==2)

    Matched                            23,866  (_merge==3)
    -----------------------------------------

*/
drop if _merge != 3
drop _merge

save ${dir}/cleaned_data/games_and_players.dta, replace
