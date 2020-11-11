rootdirectory = 'C:\Users\brank\Desktop\Petnicki Projekat\Klasifikacija Muzike\zanr_muzike';
matlabloc = fullfile(rootdirectory, 'LPCC Method');

[RAW, NUM, TXT] = xlsread('nata.xls');

path = TXT(:,1);
genre = TXT(:,2);