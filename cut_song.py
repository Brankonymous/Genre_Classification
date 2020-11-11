import pydub
import os

from pydub import AudioSegment
from pydub.audio_segment import extract_wav_headers

files_path = 'genres/full'

#Vreme pocetka i kraja seckanja sample-a
startTime = 13 * 1000
endTime = 18 * 1000


for root, dirs, files in os.walk(files_path):

    cnt = 0
    for file in files:
        #s -> string koji sadrzi ime foldera zanra
        s = ''
        for i in file:
            if (i == '.'):
                break
            s+=i

        if (file.endswith('.mf')):
            continue

        #Ekstraktovanje pesme
        song = AudioSegment.from_wav(root + '/' + file)

        for i in range(0,30,5):
            cnt += 1
            extract = song[i*1000:(i+5)*1000]
            extract.export('genres/bigger_cut/' + s + '/' + s + '_' + str(cnt) + '.wav', format = "wav")

        
        
