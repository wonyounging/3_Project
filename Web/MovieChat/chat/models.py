from django.db import models


class Member(models.Model):
    userid = models.CharField(max_length=50, null=False, primary_key=True)
    passwd = models.CharField(max_length=500, null=False)
    name = models.CharField(max_length=20, null=False)
    address = models.CharField(max_length=20, null=False)
    tel = models.CharField(max_length=20, null=True)

class Chat(models.Model):
    idx = models.AutoField(primary_key=True)
    userid = models.CharField(max_length=50, null=False)
    query = models.CharField(max_length=500, null=False)
    answer = models.CharField(max_length=1000, null=False)
    intent = models.CharField(max_length=50, null=False)

class Movie(models.Model):
    idx = models.AutoField(primary_key = True) # 순번 # 자동증가일련번호
    title = models.CharField(max_length=50, null=False) # 영화명
    opendate = models.DateTimeField(null=False) # 개봉일
    people = models.IntegerField(null=False) # 누적관객수
    grade = models.CharField(max_length=100, null=True) # 등급
    genre = models.CharField(max_length=100, null=False)  # 장르
    repnation = models.CharField(max_length=100, null=True)  # 대표국적
    nations = models.CharField(max_length=100, null=True)  # 국적
    Production = models.CharField(max_length=100, null=True)  # 제작사
    distributor = models.CharField(max_length=100, null=True)  # 배급사
    director = models.CharField(max_length=500, null=True)  # 감독
    actors = models.CharField(max_length=1000, null=True)  # 배우
    story = models.CharField(max_length=3000, null=True)  # 줄거리
    keyword = models.CharField(max_length=20, null=True)  # 키워드
    poster = models.CharField(max_length=100, null=True)  # 포스터
    totscore = models.CharField(max_length=100, null=True)  # 전체평점
    revscore = models.CharField(max_length=100, null=True)  # 리뷰평점
    review = models.CharField(max_length=2000, null=True)  # 리뷰











