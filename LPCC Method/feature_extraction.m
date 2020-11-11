rootdirectory = 'C:\Users\brank\Desktop\Petnicki Projekat\Klasifikacija Muzike\zanr_muzike';
matlabloc = fullfile(rootdirectory, 'LPCC Method');

[RAW, NUM, TXT] = xlsread('gtzan.xls');

for i = 1:length(NUM)

    cell_path = TXT(i,1);
    cell_genre = TXT(i,2);

    path = cell_path{1};
    genre = cell_genre{1};
    
    num_genre = zeros(1,10);
    
    if (genre == "blues")
        num_genre(1) = 1;
    end
    if (genre == "classical")
        num_genre(2) = 1;
    end
    if (genre == "country")
        num_genre(3) = 1;
    end
    if (genre == "disco")
        num_genre(4) = 1;
    end
    if (genre == "hiphop")
        num_genre(5) = 1;
    end
    if (genre == "jazz")
        num_genre(6) = 1;
    end
    if (genre == "metal")
        num_genre(7) = 1;
    end
    if (genre == "pop")
        num_genre(8) = 1;
    end
    if (genre == "reggae")
        num_genre(9) = 1;
    end
    if (genre == "rock")
        num_genre(10) = 1;
    end

    % sample rate natine baze = 441000
    [data, rate] = audioread(fullfile(rootdirectory,path));
    lpc_ans = lpc(data,30);
    lpc_ans = cceps(lpc_ans);

    C = cell(1,31);
    C{1,1} = genre;
    for j = 2:31
        C{1,j} = lpc_ans(j);
    end
    W = cell(1,1);
    W{1,1} = num_genre;
    Q = cell(1,30);
    for j = 1:30
        Q{1,j} = lpc_ans(j+1);
    end
    
    %writecell(C, fullfile(matlabloc,'lpcc_features_upd_gtzan.csv'), 'WriteMode','append');
    writecell(Q, fullfile(matlabloc,'lpcc_features_gtzan_in.xls'), 'WriteMode','append');
    writecell(W, fullfile(matlabloc,'lpcc_features_gtzan_out.xls'), 'WriteMode','append');

    i
end

%     cell_path = TXT(14,1);
%     cell_genre = TXT(14,2);
% 
%     path = cell_path{1};
%     genre = cell_genre{1};
% 
%     % sample rate natine baze = 441000
%     [data, rate] = audioread(fullfile(rootdirectory,path));
%     lpc_ans = lpc(data, 14);
%     
% 
%     C = cell(1,15);
%     C{1,1} = genre;
%     for j = 2:15
%         C{1,j} = lpc_ans(j);
%     end
%     
%     writecell(C, fullfile(matlabloc,'cepstrum_features.csv'), 'WriteMode','append');
