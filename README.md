
# ECE229_finalproject

**Problem & Background**

One of the most challenging problems for book lovers is to find exciting books amongst the plethora of books. When we go to the library and buy or borrow books, we risk our time, effort, and potentially money to find a book that we like. To reduce the so-called risks, we want to propose a recommendation system that helps readers choose the books that fit their needs and tastes. To do this, we want to investigate the passing trend and a combination of a book’s attributes such as genre, author and their writing experience, publishing year, etc. Also, other than filtering options, to create a more customized recommendation, the user would adjust the weight of each attribute to generate recommendations based on their preferences. 

**User story**

As a user who is looking for a new book, I want to be able to find recommendations based on my preferred genre, author, writing style, and publishing year. I want to be able to see how the book I’m reading fits into each category and to find books similar to the books I like. As a user with preferences in the book's attribute, I want to be able to sort through the library and search through books that match my descriptions. As a new reader, I want to be able to visually see the availability of genres and other book attributes and to find easy to read books based on which ones I want to try.  As an avid reader, I want to be able to input all the books I’ve read before, and to get recommendations for books I might want to try next. 


**Structure of files**

- Book-recomendation-engine (contains the Jupyter notebook for the recommendation system)
 
	- Book recommender - Varun.ipynb (code used for creating recommendation tab)
	
	- books_cleaned.csv (the cleaned dataset from book_data.csv named books_cleaned.csv)

	- further cleaning of the dataset.ipynb (includes code for capitalizing author names, ...)
	
- data
	- amazon_best_selling.csv (551 rows, 7 columns) Includes amazon best selling books
	
	- authors_dataframe_link.md (209417 rows, 20 columns) link to goodreads dataset
	
	- book_data.csv (54301 rows, 12 columns) initial goodreads books dataset
	
	- book_added_amzn.csv (54301 rows, 13 columns) goodreads books dataset augmented with a column about amazon best selling
	
	- **books_authors_final.csv** final dataset used in book search engine including books and authors information
	
	- countries_genres_freq.csv CREATED from book_data.csv and authors_df.csv dataframes containing number of genres frequencies for different countries
	
	- books_cleaned.csv (cleaned book_data.csv)
	
	
- Keyword-engine (code for the search engine with the major feature of review keywords along with countries genre distribution)

	- Keyword_prep_search.ipynb (Includes Preprocessing and book search codes)

	- Keyword Engine.ipynb (Includes code used for creating the corresponding tab for the search engine)

	- books_authors_final.csv (the final dataset used for the search tab)

- Notebooks (Includes the combined code for creating the dashboard)

	- **Dashboard.ipynb** (The final dashboard code)

	- dashboard.py (initial deployment test)
	
	- test_dashboardv3.py (dashboard test)

- Proposal_OKR.pdf (proposal and weekly OKRs)

- Slides.pdf (presented slides)

**Documentation**

Find Sphinx html builds under notebook/doc/build/html

**Testing**

```
Name                  Stmts   Miss  Cover   Missing 
---------------------------------------------------
dashboardv3.py          149     30    80%   63, 242-360, 389-391, 410-416, 424-436, 448
test_dashboardv3.py      10      0   100%
---------------------------------------------------
TOTAL                   159     30    81%
	
```

**Running our code**

We have uploaded the main Jupyter notebook which includes the code for creating our dashboard in  /notebooks/Dashboard.ipynb

STEPS TO RUN OUR NOTEBOOK:
- (1) Clone/Download our repository 
```
git clone https://github.com/sepehrfrgh/ECE229_finalproject
```
- (2) After the download has finished, 
- (3) open Terminal, and type in the command 'jupyter notebook'. a window should pop up and follow the file path to get to where your 'Dashboard.ipynb' and click to open it
- (4) make sure to pip or conda install the 3rd party modules we have listed below 
- (5) Run our Jupyter Notebook

Note: We are using this version of Python:
```
Python 3.8 
```

**Main Third-party modules we have used**

- matplotlib
- pandas
- sklearn
- numpy
- nltk
- dash
- urllib
- imageio
- string
- spacy
- math
- dash_core_components
- dash_html_components
- dash.dependencies
- dash.exceptions
- scipy

