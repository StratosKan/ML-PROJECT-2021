import urllib.request
import zipfile
url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"

file_name = "horse-or-humans.zip"
training_dir = 'horse-or-human/training/'
urllib.request.urlretrieve(url, file_name)

zip_ref = zipfile.ZipFile(file_name, 'r')
zip_ref.extractall(training_dir)
zip_ref.close()

print("Horses or Human image dataset is ready to use. Have fun!")
# downloads horse or human zip and extracts it in our project repo

