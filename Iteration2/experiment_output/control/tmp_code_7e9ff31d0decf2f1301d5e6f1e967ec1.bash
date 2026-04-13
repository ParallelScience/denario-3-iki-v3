head -n 1 data/processed_data.csv | tr ',' '
' | grep -E 'QEA_2|QEB_2|Job_Security'