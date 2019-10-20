from mongoengine import StringField, DictField, Document


class SchemaMG(Document):
    primary_label = StringField(required=True, unique=True)
    extends = StringField(default="")
    entity_schema = DictField(required=True)


if __name__ == '__main__':
    from mongodb.MongoUtil import MongoUtil
    mongoUtil = MongoUtil()
    # s = SchemaMG()
    # s.primary_label = 'Movie'
    # s.entity_schema = {'type': 'object'}
    # s.save()

    s = SchemaMG.objects(primary_label='Movie')
    print(s[0].entity_schema)
