const express = require('express');
const { check } = require('express-validator');

const eventController = require('../controllers/event-controllers');

const router = express.Router();

router.post('/fetchFromXMLAndUpdateDB', eventController.fetchAllDataFromXMLAndUpdateDB);

module.exports = router;