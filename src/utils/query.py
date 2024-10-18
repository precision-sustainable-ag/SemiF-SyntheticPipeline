import yaml
from pymongo import MongoClient
from typing import Any, Dict
from pathlib import Path
import logging
import json

from omegaconf import DictConfig, ListConfig

log = logging.getLogger(__name__)


class MongoDBQueryHandler:
    """Class to handle MongoDB queries based on configuration filters."""

    def __init__(self, cfg: DictConfig):
        """
        Initialize the MongoDB connection and load the configuration file.
        
        Args:
            db_name (str): Name of the MongoDB database.
            collection_name (str): Name of the MongoDB collection.
            config_file_path (str): Path to the YAML configuration file.
            mongo_url (str): MongoDB connection URL.
        """
        self.cfg = cfg
        self.client = MongoClient(f'mongodb://{cfg.mongodb.host}:{cfg.mongodb.port}/')
        self.db = self.client[cfg.mongodb.db]
        self.collection = self.db[cfg.mongodb.collection]
        self.output_file = Path(self.cfg.paths.resultsdir, cfg.general.project_name + ".json")
        self.query = {}

    def build_query(self) -> None:
        """
        Build the MongoDB query based on the configuration filters.
        """
        self._add_morphological_filters()
        self._add_category_filters()

    def _add_morphological_filters(self) -> None:
        """
        Add morphological filters from the configuration to the query.
        """
        morphological = self.cfg.cutout_filters.morphological

        # Handle area filter
        if 'area' in morphological and morphological['area']:
            self.query['cutout_props.area'] = {
                '$gte': morphological['area'].get('min', 0),
                '$lte': morphological['area'].get('max', float('inf'))
            }

        # Handle blur_effect filter
        self._add_range_filter(morphological, 'blur_effect', 'cutout_props.blur_effect')

        # Handle eccentricity filter
        self._add_range_filter(morphological, 'eccentricity', 'cutout_props.eccentricity')

        # Handle num_components filter
        self._add_range_filter(morphological, 'num_components', 'cutout_props.num_components')

        # Handle solidity filter
        self._add_range_filter(morphological, 'solidity', 'cutout_props.solidity')

        # Handle green_sum filter
        self._add_range_filter(morphological, 'green_sum', 'cutout_props.green_sum')

        # Handle extends_border filter
        if morphological.get('extends_border') is not None:
            self.query['cutout_props.extends_border'] = morphological['extends_border']

        # Handle is_primary filter
        if morphological.get('is_primary') is not None:
            self.query['cutout_props.is_primary'] = morphological['is_primary']

        # Handle perimeter filter
        self._add_range_filter(morphological, 'perimeter', 'cutout_props.perimeter')

    def _add_category_filters(self) -> None:
        """
        Add category filters from the configuration to the query.
        Handle cases where category fields can be exact values or lists.
        """
        # category = self.cfg.cutout_filters.category
        category = self.cfg.cutout_filters.category

        # Handle category fields that could be either exact values or lists
        for field in ['family', 'genus', 'group', 'duration', 'growth_habit', 'species', 'subclass', 'common_name']:
            if field in category and category[field]:
                value = category[field]
                # Convert ListConfig to a standard list if necessary
                if isinstance(value, ListConfig):
                    value = list(value)

                if isinstance(value, list):
                    self.query[f'category.{field}'] = {'$in': value}
                else:
                    self.query[f'category.{field}'] = value

    def _add_range_filter(self, config_section: Dict[str, Any], key: str, query_field: str) -> None:
        """
        Add a range filter (min, max) to the query if both values exist in the config.

        Args:
            config_section (Dict[str, Any]): The section of the config to read from.
            key (str): The key in the config section for the range filter.
            query_field (str): The corresponding MongoDB field to filter.
        """
        if key in config_section and config_section[key]:
            min_val = config_section[key].get('min')
            max_val = config_section[key].get('max')

            if min_val is not None or max_val is not None:
                self.query[query_field] = {}
                if min_val is not None:
                    self.query[query_field]['$gte'] = min_val
                if max_val is not None:
                    self.query[query_field]['$lte'] = max_val

    def execute_query(self) -> None:
        """Execute the query and output the results to a single JSON file."""
        documents = self.collection.find(self.query)
        total_documents = self.collection.count_documents(self.query)
        print(f"Total number of documents found: {total_documents}")
        documents_list = list(documents)
        return documents_list

        # if documents_list:
            
        #     with self.output_file.open('w') as file:
        #         json.dump(documents_list, file, default=str, indent=4)
        #     print(f"Documents output to {self.output_file}")
        # else:
        #     print("No documents found to output.")

def main(cfg: DictConfig) -> None:
    query_handler = MongoDBQueryHandler(cfg)
    query_handler.build_query()
    query_handler.execute_query()
