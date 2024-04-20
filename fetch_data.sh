#!/bin/bash

mkdir data && cd data

mkdir clean && cd clean
gdown --fuzzy https://drive.google.com/file/d/1gJ2QA4uIY6luTbj4c2h5imvB1_NkVpje/view?usp=drive_link
gdown --fuzzy https://drive.google.com/file/d/1BlAMMiZLjgoYoj3Lta4h60-LI4q-4771/view?usp=drive_link
gdown --fuzzy https://drive.google.com/file/d/1TbrVJUEJL-wTQf1dvZN9ysejvCCWlhzl/view?usp=drive_link
gdown --fuzzy https://drive.google.com/file/d/1qpXpdOWsWCLWexM8smjoR2UXu_zHiv8-/view?usp=drive_link
gdown --fuzzy https://drive.google.com/file/d/1Xrc-QdwUNIgN9PVXryWnU7H0Y1qqBOcA/view?usp=drive_link
gdown --fuzzy https://drive.google.com/file/d/1C6yg57J1MMgEN0XqjB9dxTqub-VX04fq/view?usp=drive_link
gdown --fuzzy https://drive.google.com/file/d/1mPuK05RQ6cpwM1TlGFZEL7VHAqkAkQVz/view?usp=drive_link
gdown --fuzzy https://drive.google.com/file/d/1wjqcHE6JySMrCuRa8RW_m2PbeiYh028k/view?usp=drive_link
unzip "*.zip"
rm *.zip

cd ..

# maps
gdown --fuzzy https://drive.google.com/file/d/1q9K-n6QHhz7Y55e2mGBrEvaWNiveuMpf/view?usp=sharing

# gt images, camera poses
gdown --fuzzy https://drive.google.com/file/d/1o8kKnhwCoDAckpOn0mxlVt1xRmc69VG4/view?usp=sharing

# masks for dynamic objects
gdown --fuzzy https://drive.google.com/file/d/1DjfOWRjlplTpANvS4YBebp78WtHpDgRJ/view?usp=sharing

unzip "*.zip"
rm *.zip

cd ..

mkdir pretrained && cd pretrained
# clean data weights
gdown --fuzzy https://drive.google.com/file/d/1ZjEXD1XigYyJwazdJ8PTD6uM6ell5Xyb/view?usp=sharing
# noisy data weights
gdown --fuzzy https://drive.google.com/file/d/1J_G54UECEhBivVDQMVBMo2uslv6GUr7b/view?usp=sharing

unzip "*.zip"
rm *.zip