select title.production_year from complete_cast join title on title.id = complete_cast.movie_id where movie_id < 290 order by title.production_year;
select movie_id, count(movie_id) from complete_cast group by movie_id order by movie_id LIMIT 10;
select count(movie_id) from complete_cast where movie_id < 1000;
select count(movie_id) from complete_cast where movie_id < 1000 and movie_id > 500;
select count(movie_id) from complete_cast where movie_id >= 2000 and movie_id < 3000;
select count(movie_id) from complete_cast where movie_id = 1500;
select count(movie_id) from complete_cast where movie_id != 2500;
select count(movie_id) from complete_cast where movie_id < 1000 and movie_id > 500 or movie_id = 1500;