NODE_LABEL = "Resource"
REL_TYPES = ["isLeaderOf", "isLocatedIn", "isAffiliatedTo", "owns",
             "hasGender", "wasBornIn", "isCitizenOf", "created", "diedIn",
             "happenedIn", "hasCapital", "graduatedFrom", "isPoliticianOf",
             "worksAt", "participatedIn", "hasOfficialLanguage", "imports",
             "hasNeighbor", "hasCurrency", "exports", "influences", "playsFor",
             "hasChild", "isMarriedTo", "actedIn", "directed", "hasWonPrize",
             "isConnectedTo", "isInterestedIn", "hasMusicalRole", "dealsWith",
             "wroteMusicFor", "edited"]

if __name__ == '__main__':
    json_conf = {
        "vrels": [
        ],
        "labels": [
            NODE_LABEL
        ],
        "rels": [
            # {
            #     "symmetrical": false,
            #     "type": "founder",
            #     "startLabels": [
            #         "Company"
            #     ],
            #     "transitive": false,
            #     "endLabels": [
            #         "Person"
            #     ]
            # }
        ]
    }
    for rel_type in REL_TYPES:
        json_conf['rels'].append(
            {
                "symmetrical": False,
                "type": rel_type,
                "startLabels": [
                    NODE_LABEL
                ],
                "transitive": False,
                "endLabels": [
                    NODE_LABEL
                ]
            }
        )
    with open('./viewConf.json', 'w') as f:
        import json
        json.dump(json_conf, f, indent=4)
