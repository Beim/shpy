import pandas
import csv


class Writer:

    obj = {}
    obj_rel = {}

    def get_entity_writer(self, label):
        if label not in self.obj:
            self.obj[label] = csv.writer(open('./%s.csv' % label, 'w', newline=''))
            self.obj[label].writerow([':ID', 'name'])
        return self.obj[label]

    def get_rel_writer(self, start_label, end_label):
        key = '%s:%s' % (start_label, end_label)
        if key not in self.obj_rel:
            self.obj[label] = csv.writer(open('./%s.csv' % label, 'w', newline=''))



writer = Writer()
w1 = writer.get_entity_writer('person')
w1.writerow([0, 'xixi'])

w2 = writer.get_entity_writer('person')
w2.writerow([1, 'haha'])


