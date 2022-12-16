const express = require('express');
const { check } = require('express-validator');

const usersController = require('../controllers/users-controllers');

const router = express.Router();

router.get('/', usersController.getUsers);

router.get('/:id',usersController.getUser);

router.delete('/:id', usersController.deleteUser)

router.post(
  '/signup',
  [
    check('name')
      .not()
      .isEmpty(),
    check('password').isLength({ min: 3 })
  ],
  usersController.signup
);

router.post('/login', usersController.login);

module.exports = router;
