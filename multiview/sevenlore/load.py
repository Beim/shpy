import csv
import pandas

sevenlore_csv_path = r"D:\Documents\KG\7Lore_triple.csv"

if __name__ == '__main__':
    # with open(sevenlore_csv_path, 'r', encoding='utf-8') as csvfile:
    #     spamreader = csv.reader(csvfile, delimiter=',')
    #     idx = 0
    #     for row in spamreader:
    #
    #         idx += 1
    #         print(row)
    #         # print(''.join(row))
    #         if idx > 10:
    #             break
    df = pandas.read_csv(sevenlore_csv_path)
    
    print(df)

