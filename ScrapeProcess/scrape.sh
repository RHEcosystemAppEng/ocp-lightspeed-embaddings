#!/bin/bash

# openshift container platform
# 4.13
# https://access.redhat.com/documentation/en-us/openshift_container_platform/4.13/html-single/installing/index#ocp-installation-overview
if [[ $# == 2 ]]
then
  product=$(echo ${1// /_} | tr [[:upper:]] [[:lower:]])
  version=$2
  main_url="https://access.redhat.com/documentation/en-us/${product}/${version}"
else
  main_url=$1
  product=$(echo $main_url | cut -d/ -f 6)
  version=$(echo $main_url | cut -d/ -f 7)
fi

urls=$(curl $main_url 2> /dev/null | grep 'Single-page HTML' | cut -d'"' -f 2 | awk '{print "https://access.redhat.com" $1}')

mkdir scrape 2>/dev/null || true

# Prepare scrape directory by deleting the potentially conflicting files
rm -f scrape/*.html
rm -f scrape/sitemap.json

echo '{' > scrape/sitemap.json

notfirst=0

for url in $urls
do
  if [[ $notfirst == 1 ]]
  then
    echo "," >> scrape/sitemap.json
  fi
  notfirst=1
  curl -L $url -o scrape/$(echo $url | awk '{ n = split($0, a, "/"); print a[n] }').html
  echo "\"$(echo $url | awk '{ n = split($0, a, "/"); print a[n] }')\": \"$url\" " >> scrape/sitemap.json
done

echo '}' >> scrape/sitemap.json

#./embed-remote.py "${product}-${version//./_}"

