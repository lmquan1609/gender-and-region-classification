function Downloader(){
    python download_gdrive.py 10YJHvRUwphj9K6-Chj-Pm3Cfs_I35eOD training128mel.pkl

    python download_gdrive.py 1zUgvp4xNNpJzxM_9-e_b606qlukdNTM8 validation128mel.pkl

    python download_gdrive.py 1H1_8nUVdq2qlK6D26UHIEp-dY1n8-txx testing128mel.pkl
}

function Cleanup(){
    mkdir -p data/3seconds/store_spectrograms

    mv *.pkl data/3seconds/store_spectrograms
}

function Main(){
    Downloader
    Cleanup
}

Main
exit 0