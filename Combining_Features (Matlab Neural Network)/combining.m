rootdirectory = 'C:\Users\brank\Desktop\Petnicki Projekat\Klasifikacija Muzike\zanr_muzike';
matlabloc = fullfile(rootdirectory, '\Combining_Features\GTZAN');

[RAW_FGP, NUM_FGP, TXT_FGP] = xlsread('FGP_feature_gtzan.xls');
[RAW_LOG, NUM_LOG, TXT_LOG] = xlsread('LOG_feature_gtzan.xls');
[RAW_SPEC, NUM_SPEC, TXT_SPEC] = xlsread('SPEC_feature_gtzan.xls');
[RAW_LPCC, NUM_LPCC, TXT_LPCC] = xlsread('LPCC_feature_gtzan.xls');
[RAW_MFCC, NUM_MFCC, TXT_MFCC] = xlsread('MFCC_feature_gtzan.xls');

for i = 1:length(NUM_LOG)
    feature_fgp = zeros(1,10);
    feature_log = zeros(1,32);
    feature_spec = zeros(1,32);
    feature_spc = zeros(1,32);
    feature_lpcc = zeros(1,30);
    feature_mfcc = zeros(1,20);
    
    for j = 3:12
        feature_fgp(j-2) = cell2mat(TXT_FGP(i,j)) * 10;
    end
    
    for j = 2:33
        feature_log(j-1) = cell2mat(TXT_LOG(i,j));
        feature_spec(j-1) = cell2mat(TXT_SPEC(i,j)) / 1000000;
    end 
    
    for j = 1:30
        feature_lpcc(j) = cell2mat(TXT_LPCC(i,j));
    end
    
    for j = 1:20
        feature_mfcc(j) = cell2mat(TXT_MFCC(i,j)) / 1000;
    end
    
    feature_spc = horzcat(feature_log, feature_spec);
    comb_features = horzcat(feature_fgp);
    
    Q = cell(1,length(comb_features));
    for j = 1:length(comb_features)
        Q{1,j} = comb_features(j);
    end
    
    writecell(Q, fullfile(matlabloc,'FGP_feature_gtzan.csv'), 'WriteMode','append');
    
    i
end