import os
import pandas as pd

from autoviz.AutoViz_Class import AutoViz_Class
from io import BytesIO
from uuid import uuid4

from Storage import MinIO

av = AutoViz_Class()


def visualization_df(minio: MinIO, df: pd.DataFrame, job_id: str):
    if not minio.bucket_exists("diastemaviz"):
        minio.make_bucket("diastemaviz")
    
    visualization_id = uuid4().hex
    plot_dir = f"/tmp/{visualization_id}"
    plot_av_dir = os.path.join(plot_dir, "AutoViz")

    av.AutoViz(
        filename="",
        sep=",",
        depVar="",
        dfte=df,
        header=0,
        verbose=2,
        lowess=False,
        chart_format="html",
        max_rows_analyzed=150000,
        max_cols_analyzed=30,
        save_plot_dir=plot_dir,
    )

    for f in os.listdir(plot_av_dir):
        if not f.endswith(".html"):
            continue

        with open(os.path.join(plot_av_dir, f), "r") as plot_file:
            plot = plot_file.read().encode("utf-8")
            plot_size = len(plot)
            plot = BytesIO(plot)
            plot.seek(0)
            minio.put_object("diastemaviz", f"{job_id}/visualization_{visualization_id}_{f}", plot, plot_size)


def visualization(minio: MinIO, input_bucket: str, input_path: str, job_id: str):
    response = minio.get_object(input_bucket, input_path).read()
    buffer = BytesIO(response)
    buffer.seek(0)

    df = pd.read_csv(buffer)
    visualization_df(minio, df, job_id)
