const express = require('express');
const { check } = require('express-validator');

const usersController = require('../controllers/users-controllers');
const adminController = require('../controllers/admin-controllers');

const router = express.Router();

router.get('/events', adminController.getEvents);

router.get('/location',adminController.getLocations);

router.get('/:id',adminController.getEvent);


router.post(
  '/',
  [
    check('date')
    .isDate()
    //...etc
  ],
  adminController.createEvent
);

router.patch('/:id',adminController.updateEvent)

router.delete('/:id',adminController.deleteEvent)

module.exports = router;
