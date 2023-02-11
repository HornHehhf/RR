import spacy
import json
import time
from decimal import Decimal
import pywikibot
from pywikibot import exception, ItemPage, Site, PropertyPage

site = pywikibot.Site('wikidata', 'wikidata')
nlp = spacy.load("en_core_web_md")
nlp.add_pipe("entityLinker", last=True)


def entity_linking(sentence):
    doc = nlp(sentence)
    # returns all entities in the whole document
    all_linked_entities = doc._.linkedEntities
    # iterates over sentences and prints linked entities
    entities = []
    for entity in all_linked_entities:
        print(entity.get_url().split('/')[-1], entity.get_label(), entity.get_description(), str(entity.get_span()))
        entities.append((entity.get_url().split('/')[-1], entity.get_label(), entity.get_description(),
                         str(entity.get_span())))
    return entities


property_id_label = {}
item_id_label = {}


def get_propertyid_label(site, claim_id):
    if claim_id not in property_id_label:
        property_page = PropertyPage(site, claim_id)
        name = property_page.get()['labels']['en']
        property_id_label[claim_id] = name
    return property_id_label[claim_id]


def get_itemid_label(site, claim_id):
    if claim_id not in item_id_label:
        property_page = ItemPage(site, claim_id)
        item_id_label[claim_id] = property_page.get()
    return item_id_label[claim_id]['labels']['en']


def get_quantity(quantity):
    if isinstance(quantity.amount, Decimal):
        quantity.amount = str(quantity.amount)
    if quantity.unit != '1':
        return quantity.amount, get_itemid_label(site, quantity.unit.split('/')[-1])
    return quantity.amount


def get_wikidata(wiki_id):
    temporal_relations = []
    if wiki_id not in item_id_label:
        try:
            item_id_label[wiki_id] = pywikibot.ItemPage(site, wiki_id).get()
        except:
            return temporal_relations
    clm_dict = item_id_label[wiki_id]['claims']
    for key in clm_dict.keys():
        clm_list = clm_dict[key]
        for clm in clm_list:
            predicate = key
            obj = clm.getTarget()
            other_roles = []
            qualifiers = clm.qualifiers
            for qualifier in qualifiers.keys():
                for claim in qualifiers[qualifier]:
                    claim_id = claim.getID()
                    target = claim.getTarget()
                    if isinstance(target, pywikibot.WbTime):
                        other_roles.append((claim_id, target))
            if isinstance(obj, pywikibot.WbTime) or len(other_roles) > 0:
                if isinstance(obj, pywikibot.page._filepage.FilePage) or obj is None:
                    continue
                relation_data = {}
                print('\tpredicate:', get_propertyid_label(site, predicate))
                relation_data['predicate'] = get_propertyid_label(site, predicate)
                if isinstance(obj, pywikibot.page._wikibase.ItemPage):
                    obj_id = obj.getID()
                    if obj_id not in item_id_label:
                        item_id_label[obj_id] = obj.get()
                    try:
                        print('\tobject:', item_id_label[obj_id]['labels']['en'])
                        relation_data['object'] = item_id_label[obj_id]['labels']['en']
                    except KeyError:
                        continue
                elif isinstance(obj, pywikibot.WbTime):
                    print('\tobject:', obj.year)
                    relation_data['object'] = obj.year
                elif isinstance(obj, pywikibot.WbQuantity):
                    print('\tobject:', get_quantity(obj))
                    relation_data['object'] = get_quantity(obj)
                elif isinstance(obj, pywikibot.WbMonolingualText):
                    print('\tobject:', obj.text)
                    relation_data['object'] = obj.text
                elif isinstance(obj, str):
                    print('\tobject:', obj)
                    relation_data['object'] = obj
                else:
                    print(type(obj))
                    print('\tobject:', obj.toWikibase())
                    relation_data['object'] = obj.toWikibase()
                for role in other_roles:
                    print('\t' + str(get_propertyid_label(site, role[0])) + ":", role[1].year)
                    relation_data[get_propertyid_label(site, role[0])] = role[1].year
                print()
                temporal_relations.append(relation_data)
    return temporal_relations


def get_linked_entities(test_path, output_path, option):
    output_option = 'entity_linking'
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    for idx, test_case in enumerate(test_data):
        print(idx, len(test_data))
        predictions = test_case[option]
        test_case[output_option] = []
        for prediction in predictions:
            prediction = str(prediction).strip()
            nlp_text = nlp(prediction).sents
            sent_entities = []
            for sent in nlp_text:
                sent = str(sent)
                entities = entity_linking(sent)
                sent_entities.append((sent, entities))
            test_case[output_option].append(sent_entities)
            with open(output_path, 'w') as f:
                json.dump(test_data, f, indent=4)


def get_temporal_knowledge(test_path, knowledge_path):
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    temporal_data = {}
    visited_entities = set()
    for idx, test_case in enumerate(test_data):
        print(idx, len(test_data))
        entity_predictions = test_case['entity_linking']
        for sent_entities in entity_predictions:
            for sent, entities in sent_entities:
                for entity in entities:
                    wiki_id = entity[0]
                    if wiki_id not in temporal_data and wiki_id not in visited_entities:
                        print(entity[1])
                        temporal_relations = get_wikidata(entity[0])
                        visited_entities.add(entity[0])
                        if len(temporal_relations) > 0:
                            temporal_data[wiki_id] = temporal_relations
                            with open(knowledge_path, 'w') as f:
                                json.dump(temporal_data, f, indent=4)


def covert_relation_to_sentence(entity, temporal_relations):
    # Only Consider templates that appear more than 1000 times
    templates = dict()
    kb_candidates = []
    for relation in temporal_relations:
        for key, value in relation.items():
            if isinstance(value, list):
                relation[key] = ' '.join(value)
            elif isinstance(value, dict):
                relation[key] = str(int(value['latitude'])) + ', ' + str(int(value['longitude']))
            else:
                relation[key] = str(value)
        template_key = tuple(sorted(relation.keys()))
        if template_key not in templates:
            templates[template_key] = 1
        else:
            templates[template_key] += 1
        if template_key == tuple(sorted(['predicate', 'object', 'point in time'])):
            # template for (predicate, object, point in time):
            # The (predicate) of (entity) was (object) in (point in time).
            sentence = 'The ' + relation['predicate'] + ' of ' + entity[1] + ' was ' + relation['object'] \
                       + ' in ' + relation['point in time'] + '.'
            kb_candidates.append(sentence)
        elif template_key == tuple(sorted(['predicate', 'object', 'start time'])):
            # template for (predicate, object, start time):
            # (entity) (predicate) (object) since (start time).
            sentence = entity[1] + ' ' + relation['predicate'] + ' ' + relation['object'] \
                       + ' since ' + relation['start time'] + '.'
            kb_candidates.append(sentence)
        elif template_key == tuple(sorted(['predicate', 'object', 'start time', 'end time'])):
            # template for (predicate, object, start time, end time):
            # (entity) (predicate) (object) from (start time) to (end time).
            sentence = entity[1] + ' ' + relation['predicate'] + ' ' + relation['object'] \
                       + ' from ' + relation['start time'] + ' to ' + relation['end time'] + '.'
            kb_candidates.append(sentence)
        elif template_key == tuple(sorted(['predicate', 'object'])):
            # template for (predicate, object):
            # The (predicate) of (entity) was (object).
            sentence = 'The ' + relation['predicate'] + ' of ' + entity[1] + ' was ' + relation['object'] + '.'
            kb_candidates.append(sentence)
        elif template_key == tuple(sorted(['predicate', 'object', 'publication date'])):
            # template for (predicate, object, publication date):
            # The (predicate) of (entity) was (object) since (publication date).
            sentence = 'The ' + relation['predicate'] + ' of ' + entity[1] + ' was ' + relation['object'] \
                       + ' since ' + relation['publication date'] + '.'
            kb_candidates.append(sentence)
        elif template_key == tuple(sorted(['predicate', 'object', 'end time'])):
            # template for (predicate, object, end time):
            # (entity) (predicate) (object) until (end time).
            sentence = entity[1] + ' ' + relation['predicate'] + ' ' + relation['object'] \
                       + ' until ' + relation['end time'] + '.'
            kb_candidates.append(sentence)
        elif template_key == tuple(sorted(['predicate', 'object', 'start time', 'position in time'])):
            # template for (predicate, object, start time, point in time):
            # The (predicate) of (entity) was (object) since (start time).
            sentence = 'The ' + relation['predicate'] + ' of ' + entity[1] + ' was ' + relation['object'] \
                       + ' since ' + relation['start time'] + '.'
            kb_candidates.append(sentence)
        elif template_key == tuple(sorted(['predicate', 'object', 'inception', 'dissolved, abolished or demolished date'])):
            # template for (predicate, object, inception, dissolved_abolished_or_demolished_date):
            # (entity) (predicate) (object) from (inception) to (dissolved, abolished or demolished date)
            sentence = entity[1] + ' ' + relation['predicate'] + ' ' + relation['object'] \
                       + ' from ' + relation['inception'] + ' to ' \
                       + relation['dissolved, abolished or demolished date'] + '.'
            kb_candidates.append(sentence)
    return templates, kb_candidates


def get_kb_candidates(test_path, knowledge_path, output_path):
    test_file = open(test_path, 'r')
    test_data = json.load(test_file)
    knowledge_file = open(knowledge_path, 'r')
    temporal_data = json.load(knowledge_file)
    relation_templates = dict()
    template_example = dict()
    for idx, test_case in enumerate(test_data):
        print(idx, len(test_data))
        entity_predictions = test_case['entity_linking']
        candidate_predictions = []
        for sent_entities in entity_predictions:
            sent_candidates = []
            for sent, entities in sent_entities:
                cur_candidates = set()
                for entity in entities:
                    wiki_id = entity[0]
                    if wiki_id in temporal_data:
                        cur_templates, kb_candidates = covert_relation_to_sentence(entity, temporal_data[wiki_id])
                        for template, num in cur_templates.items():
                            if template not in relation_templates:
                                relation_templates[template] = num
                                for relation in temporal_data[wiki_id]:
                                    template_key = tuple(sorted(relation.keys()))
                                    if template_key == template:
                                        template_example[template] = ({'entity': entity[1]}, relation)
                            else:
                                relation_templates[template] += num
                        for sentence in kb_candidates:
                            cur_candidates.add(sentence)
                sent_candidates.append((sent, list(cur_candidates)))
            candidate_predictions.append(sent_candidates)
        del test_case['entity_linking']
        test_case['kb_candidates'] = candidate_predictions
    relation_templates = sorted(relation_templates.items(), key=lambda item: item[1], reverse=True)
    for template, num in relation_templates:
        print(template, num)
        print(template_example[template])
        print()
    with open(output_path, 'w') as f:
        json.dump(test_data, f, indent=4)


if __name__ == '__main__':
    dir_path = '/path/to/working/dir/Temporal/'
    self_consistency_gpt3_path = dir_path + 'GPT-3/implicit_temporal_self_consistency_gpt3.json'
    self_consistency_entity_path = dir_path + 'GPT-3/implicit_temporal_self_consistency_gpt3_entities.json'
    temporal_data_path = dir_path + '/GPT-3/implicit_temporal_self_consistency_gpt3_wikidata.json'
    self_consistency_kb_path = dir_path + 'GPT-3/implicit_temporal_self_consistency_gpt3_kb_candidates.json'
    time_start = time.time()
    get_linked_entities(self_consistency_gpt3_path, self_consistency_entity_path, option='self_consistency_gpt3')
    get_temporal_knowledge(self_consistency_entity_path, temporal_data_path)
    get_kb_candidates(self_consistency_entity_path, temporal_data_path, self_consistency_kb_path)
    time_end = time.time()
    print('time:', time_end - time_start)

