pairs_tr = [(0, 'ETL7'), (1, 'ETL9G')]
pairs_te = [(0, 'ETL7'), (1, 'ETL9G')]

label2wid_tr = {k:v for k, v in pairs_tr}
label2wid_te = {k:v for k, v in pairs_te}

wid2label_tr = {v:k for k, v in pairs_tr}
wid2label_te = {v:k for k, v in pairs_te}
