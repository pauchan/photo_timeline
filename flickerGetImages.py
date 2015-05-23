import urllib2
import json
import os
import time

request = "https://api.flickr.com/services/rest/?"
#request_dict = {'nojsoncallback': '1',
#                'method': 'flickr.photos.search',
#                'format': 'json',
#                'per_page': 30,privacy_filter=1,
#                'api_key': 'bb172cc50015aa36d7354a3a2ec84dde'}

start_year = 1914
end_year = 2000

for year in range(start_year,end_year):
    start_time = time.time()
    if not os.path.exists('trainingimages/{}'.format(year)):
        os.makedirs('trainingimages/{}'.format(year))
    full_request = request + 'is_commons=true&nojsoncallback=1&method=flickr.photos.search&format=json&per_page=100&privacy_filter=1&api_key=bb172cc50015aa36d7354a3a2ec84dde&min_taken_date={}-01-01&max_taken_date={}-12-31'.format(year,year)
    request_data = urllib2.urlopen(full_request).read()
    jsonData = json.loads(request_data)
    photos_dict = jsonData['photos']
    photo_array = photos_dict['photo']
    for idx,photo in enumerate(photo_array):
        url = 'https://farm{}.staticflickr.com/{}/{}_{}_s.jpg'.format(photo['farm'],photo['server'],photo['id'],photo['secret'])
        imageData = urllib2.urlopen(url).read()

        f = open('trainingimages/{}/{}.png'.format(year,idx),'w+')
        f.write(imageData)
        f.close()
        print url
    stop_time = time.time()
    print 'year {} pictures downloaded: {} execution time: {}'.format(year, stop_time-start_time, len(photo_array))
