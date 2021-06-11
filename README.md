# ECE229_finalproject

**Problem & Background**

One of the most challenging problems for book lovers is to find exciting books amongst the plethora of books. When we go to the library and buy or borrow books, we risk our time, effort, and potentially money to find a book that we like. To reduce the so-called risks, we want to propose a recommendation system that helps readers choose the books that fit their needs and tastes. To do this, we want to investigate the passing trend and a combination of a book’s attributes such as genre, author and their writing experience, publishing year, etc. Also, other than filtering options, to create a more customized recommendation, the user would adjust the weight of each attribute to generate recommendations based on their preferences. 


**Structure of files**

- Book-recomendation-engine

- data
	- amazon_best_selling.csv (551 rows, 7 columns) Includes amazon best selling books
	
	- authors_dataframe_link.md (209417 rows, 20 columns) link to goodreads dataset
	
	- book_data.csv (54301 rows, 12 columns) initial goodreads books dataset
	
	- book_added_amzn.csv (54301 rows, 13 columns) goodreads books dataset augmented with a column about amazon best selling
	
	- **books_authors_final.csv** final dataset used in book search engine including books and authors information
	
	- countries_genres_freq.csv CREATED from book_data.csv and authors_df.csv dataframes containing number of genres frequencies for different countries
	
	- data_isbn.csv MODIFIED by us ?
	
- Keyword-engine

- Notebooks

- Proposal_OKR.pdf (proposal and weekly OKRs)

- Slides.pdf (presented slides)

