class DataFrameCleaner:
    def drop_missing(self, axis=0, how='any'):
        print(f"Dropping missing with axis={axis}, how={how}")

    def fill_missing(self, strategy='mean'):
        print(f"Filling missing with strategy={strategy}")

    def drop_duplicates(self):
        print("Dropping duplicates...")

    def reset_index(self):
        print("Resetting index...")

    def to_csv(self, path='output.csv'):
        print(f"Exporting to CSV: {path}")
