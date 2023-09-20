import click
import pandas as pd

CLASS_MAP = {k: v for v, k in {
    'class 1: rod': 1,
    'class 2: RBC/WBC': 2,
    'class 3: yeast': 3,
    'class 4: misc': 4,
    'class 5: single EPC': 5,
    'class 6: few EPC ': 6,
    'class 7: several EPC': 7
    }.items()
}

@click.command()
@click.argument('csv_path', type=click.Path(exists=True))
def main(csv_path):
    pdf = pd.read_csv(csv_path, index_col=0)
    pdf['label'] = pdf['Class'].map(lambda l: CLASS_MAP[l])
    pdf = pdf.drop('Class', axis=1)
    pdf.to_csv(csv_path)

if __name__ == '__main__':
    main()
