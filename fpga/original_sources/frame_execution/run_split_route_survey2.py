"""Retry frozen route analysis with the separately approved six-GiB Job."""
import survey_split_output_routes as survey
from run_units import sha

assert sha(survey.HERE/'survey_split_output_routes.py') == '3f691717b1061ee04a3fd5ad208d018d36a5558e659d421c93211dba71cc3818'
survey.RUN=survey.BASE/'split-outputs-route-survey2'
if __name__=='__main__':survey.main()
