# An anime recommender system based on content using tree models

## Introduction

This was the nanodegree's capstone project of machine learning engineering at Udacity, choosing a project that I wanted develop. Therefore I decided to build a system of anime recommendations, moving away from topics more related to logistics or finance (which I have been studying in my background and developing in my work), thus broadening the horizons.

In this project, I seek to develop a content-based system that can predict the possible ranking that a user will give to a given anime, propose new series, and movies of whatever genre they can enjoy.

## About Data

In this project, we have two datasets that are taken from Kaggle. In case you want to review the source, The datasets can obtain it from the following link: [https://www.kaggle.com/CooperUnion/anime-recommendations-database]
This data set contains information on user preference data from 73,516 users on 12,294 anime from myanimelist.net. Each user can add anime to their completed list and give it a rating, and this data set is a compilation of those ratings.
Anime.csv

* anime_id - myanimelist.net's unique id identifying an anime.
* name - full name of anime.
* genre - comma separated list of genres for this anime.
* type - movie, TV, OVA, etc.
* episodes - how many episodes in this show. (1 if movie).
* rating - average rating out of 10 for this anime.
* members - number of community members that are in this anime's
 "group".

The second dataset is Rating.csv

* user_id - non-identifiable randomly generated user id.
* anime_id - the anime that this user has rated.
* rating - rating out of 10 this user has assigned (-1 if the user watched it but did not give a rating).

## Technology

The Python version in this project is **3.8.5**,  the libraries and their versions are the following:

1. pandas: 1.1.3
2. numpy: 1.19.2
3. matplotlib: 3.3.2
4. scikit-learn: 0.23.2
5. seaborn: 0.11.0

## Main Results and future work

The metric based on this project was **MAE** (mean absolute error), and the model used as a benchmark was a linear regression, so it was relatively easy to beat it. Algorithms of decision trees and random forest were used. In later work, ensemble methods or XGboost may be considered. Better solutions could also be explored with Deep Learning.
