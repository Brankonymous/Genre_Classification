rootdirectory = 'C:\Users\brank\Desktop\Petnicki Projekat\Klasifikacija Muzike\zanr_muzike';
matlabloc = fullfile(rootdirectory, '\Combining_Features');

[RAW, NUM, TXT] = xlsread('gtzan.xls');

for i = 1:length(NUM)
    cell_path = TXT(i,1);
    cell_genre = TXT(i,2);

    path = cell_path{1};
    genre = cell_genre{1};
    
    num_genre = zeros(1,6);
    
    if (genre == "classical")
        num_genre(1) = 1;
    end
    if (genre == "folk")
        num_genre(2) = 1;
    end
    if (genre == "house")
        num_genre(3) = 1;
    end
    if (genre == "jazz")
        num_genre(4) = 1;
    end
    if (genre == "rnb")
        num_genre(5) = 1;
    end
    if (genre == "rock")
        num_genre(6) = 1;
    end
    
    % sample rate natine baze = 441000
    [data, rate] = audioread(fullfile(rootdirectory,path));
    cep = cceps(data);
    cepstrum = zeros(1,30);
    for j = 0:29
        sum = 0;
        for k = 1:11025
            sum = sum + cep(11025*j + k);
        end
        sum = sum / 11025;
        cepstrum(j+1) = sum;
    end
    
    Q = cell(1,15);
    for j = 1:15
        Q{1,j} = cepstrum(j);
    end
    W = cell(1,6);
    for j = 1:6
        W{1,j} = num_genre(j);
    end
    
    %writecell(C, fullfile(matlabloc,'cesptrum_features_upd_gtzan.csv'), 'WriteMode','append');
    writecell(Q, fullfile(matlabloc,'cesptrum_features_gtzan_in.csv'), 'WriteMode','append');
    writecell(W, fullfile(matlabloc,'cesptrum_features_gtzan_out.csv'), 'WriteMode','append');
    
    i
end