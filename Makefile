NAME	=	my_torch_analyzer

all:	$(NAME)

$(NAME):
	chmod +x $(NAME)

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

fclean:	clean

re:	fclean all

tests_run:
	python3 -m pytest -q

.PHONY: all $(NAME) clean fclean re tests_run
