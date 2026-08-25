import sqlite3
import logging
from datetime import datetime
import hydra
import omegaconf
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class SQLiteQueryHandler:
    """
    A class to build and execute SQL queries dynamically.
    
    This class lets you add conditions, build the query string, execute it,
    and then format or save the results.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the QueryBuilder with a SQLite database file and target table.
        
        Args:
            db_path (str): Path to the SQLite database file.
            table_name (str): Name of the table on which to run queries.
        """
        self.db_path = cfg.paths.sql_database
        self.table_name = cfg.sqlite3.cutouts
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        self.conditions = []  # List to hold SQL conditions (strings).
        self.params = []      # List to hold values corresponding to the conditions.
        log.info("Connected to database: %s", self.db_path)
        self.filter = cfg.cutout_filters

    def add_conditions(self) -> None:
        """
        Add conditions to the query based on the configuration filters.
        """
        self.add_morphological_condition()
        self.add_category_condition()
        self.add_non_target_weed_condition()
        self.add_validated_filter()
    
    def add_condition(self, column: str, operator: str, value) -> None:
        """
        Add a condition to the query.

        This method appends a condition (with a parameter placeholder) to the list
        of conditions and saves the parameter value. For example:
            add_condition("age", ">", 30)
        will later add "age > ?" to the query and 30 as a parameter.
        
        Args:
            column (str): The column name.
            operator (str): SQL operator (e.g., '=', '>', '<', 'LIKE', etc.).
            value: The value to compare.
        """
        condition = f"{column} {operator} ?"
        self.conditions.append(condition)
        self.params.append(value)
        log.info("Added condition: %s with value: %s", condition, value)


    def build_query(self) -> str:
        """
        Construct the SQL query string from the base query and any conditions.
        
        Returns:
            str: The full SQL query.
        """
        base_query = f"SELECT * FROM {self.table_name}"
        if self.conditions:
            where_clause = " AND ".join(self.conditions)
            query = f"{base_query} WHERE {where_clause}"
        else:
            query = base_query
        log.info("Built query: %s", query)
        return query

    def execute_query(self) -> tuple[list, list]:
        """
        Execute the constructed SQL query and return the results.
        
        Returns:
            tuple: A tuple containing the list of rows and a list of column names.
        """
        query = self.build_query()
        start_time = datetime.now()
        try:
            log.info("Executing query: %s with parameters: %s", query, self.params)
            self.cursor.execute(query, self.params)
            rows = self.cursor.fetchall()
            col_names = [desc[0] for desc in self.cursor.description]
            log.info("Query executed in: %s", datetime.now() - start_time)
            return rows, col_names
        except sqlite3.Error as e:
            log.error("Query execution failed: %s", e)
            return [], []


    def close(self) -> None:
        """Commit changes (if any) and close the database connection."""
        self.connection.commit()
        self.connection.close()
        log.info("Database connection closed.")

    
    def add_bbox_area_condition(self) -> None:
        bbox_config = self.filter.morphological.get('bbox_area_cm2')
        if not bbox_config:
            return

        # Get default values
        default_range = bbox_config.get('default', {})
        default_min = default_range.get('min', 0)
        default_max = default_range.get('max', float('inf'))

        # Check for common name specific overrides
        cn_ranges = bbox_config.get('common_name_ranges', {})

        if cn_ranges:
            conditions = []
            parameters = []

            # For each common name, add a sub-condition
            for cn, range_vals in cn_ranges.items():
                cn_lower = cn.lower().strip()
                min_val = range_vals.get('min', default_min)
                max_val = range_vals.get('max', default_max)
                conditions.append("(LOWER(trim(category_common_name)) = ? AND estimated_bbox_area_cm2 >= ? AND estimated_bbox_area_cm2 <= ?)")
                parameters.extend([cn_lower, min_val, max_val])

            # For records whose common name is not specified in the overrides, apply the default range.
            placeholders = ", ".join("?" for _ in cn_ranges.keys())
            default_condition = f"(LOWER(trim(category_common_name)) NOT IN ({placeholders}) AND estimated_bbox_area_cm2 >= ? AND estimated_bbox_area_cm2 <= ?)"
            # Add the lower-cased common names to the parameters.
            parameters.extend([cn.lower().strip() for cn in cn_ranges.keys()])
            parameters.extend([default_min, default_max])
            conditions.append(default_condition)

            # Combine all sub-conditions with OR.
            full_condition = "(" + " OR ".join(conditions) + ")"
            self.conditions.append(full_condition)
            self.params.extend(parameters)
        else:
            # If no specific common name ranges, fall back to default
            self.add_condition("estimated_bbox_area_cm2", ">=", default_min)
            self.add_condition("estimated_bbox_area_cm2", "<=", default_max)

    def add_validated_filter(self) -> None:
        """
        Add a filter for the 'validated' column.
        """
        # Assume the filter value is provided in the configuration under self.filter.validated
        validated_value = self.filter.get("validated", None)
        if validated_value is not None:
            self.add_condition("validated", "=", validated_value)
            
    def add_range_filter(self, morph: dict, key: str, column: str) -> None:
        """
        Add a range filter to the query.
        
        Args:
            morph (dict): The morphological filter dictionary.
            key (str): The key in the morphological filter dictionary.
            column (str): The column name in the database.
        """
        if key in morph and morph[key]:
            min_val = morph[key].get('min', 0)
            max_val = morph[key].get('max', float('inf'))
            if min_val is not None or max_val is not None:
                if min_val != float('inf') or min_val is not None:
                    self.add_condition(column, ">=", min_val)
                if max_val != float('inf') or max_val is not None:
                    self.add_condition(column, "<=", max_val)
    
    def add_morphological_condition(self) -> None:
        """
        Add morphological filters from the configuration to the query.
        """
        morph = self.filter.morphological

        # Handle area filter
        # if 'bbox_area_cm2' in morph and morph.bbox_area_cm2:
        #     self.add_condition("json_extract(cutout_props, '$.bbox_area_cm2')", ">=", morph.bbox_area_cm2.get('min', 0))
        #     self.add_condition("json_extract(cutout_props, '$.bbox_area_cm2')", "<=", morph.bbox_area_cm2.get('max', float('inf')))
        self.add_bbox_area_condition()
        # Handle extends_border filter
        if morph.get('extends_border') is not None:
            self.add_condition("extends_border", "=", morph.extends_border)

        # Handle is_primary filter
        if morph.get('is_primary') is not None:
            self.add_condition("is_primary", "=", morph.is_primary)

        # Handle blur_effect filter
        self.add_range_filter(morph, 'blur_effect', 'blur_effect')

        # Handle num_components filter
        self.add_range_filter(morph, 'num_components', 'num_components')

    def add_category_condition(self) -> None:
        """
        Add category filters from the configuration to the query.
        Handle cases where category fields can be exact values or lists.
        """
        category = self.filter.category
        for key, value in category.items():
            # Only process if there is a value provided.
            if value:
                if key == 'common_name':
                    # Handle common_name case-insensitively.
                    # If the value is a list or ListConfig, process it as a list.
                    if isinstance(value, (list, omegaconf.listconfig.ListConfig)):
                        names = list(value)  # Convert to a regular list if necessary.
                        # Build placeholders for each name.
                        placeholders = ", ".join("?" for _ in names)
                        # Construct the condition:
                        # LOWER(json_extract(category, '$.common_name')) IN (?, ?, ...)
                        condition = f"LOWER(trim(category_common_name)) IN ({placeholders})"
                        # condition = "LOWER(trim(json_extract(category, '$.common_name'))) = ?"
                        self.conditions.append(condition)
                        # Append the lower-cased names to the parameters.
                        self.params.extend([name.lower().strip() for name in names])
                    else:
                        # If value is a single string, use an equality check.
                        # condition = "LOWER(json_extract(category, '$.common_name')) = ?"
                        condition = "LOWER(trim(category_common_name)) = ?"
                        self.conditions.append(condition)
                        self.params.append(value.lower().strip())
                else:
                    # For other category keys, apply a simple equality filter.
                    condition = f"category_{key} = ?"
                    self.conditions.append(condition)
                    self.params.append(value)
    def add_non_target_weed_condition(self) -> None:
        """
        Add filters for non-target weeds from the configuration to the query.
        """
        non_target_weed = self.filter.morphological.non_target_weed
        non_target_weed_pred_conf = self.filter.morphological.non_target_weed_pred_conf

        # Handle non_target_weed filters either True or False
        if non_target_weed is not None:
            self.add_condition("non_target_weed", "=", non_target_weed)

        # Handle non_target_weed_pred_conf filter which has a min and max value
        if non_target_weed_pred_conf.min or non_target_weed_pred_conf.max:
            min_conf = non_target_weed_pred_conf.get('min', 0.0)
            max_conf = non_target_weed_pred_conf.get('max', 1.0)
            self.add_condition("non_target_weed_pred_conf", ">=", min_conf)
            self.add_condition("non_target_weed_pred_conf", "<=", max_conf)

@hydra.main(version_base="1.2", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    
    # Usage example:
    qb = SQLiteQueryHandler(cfg)
    qb.add_conditions()
    rows, columns = qb.execute_query()
    qb.close()

# Example usage:
if __name__ == "__main__":
    main()